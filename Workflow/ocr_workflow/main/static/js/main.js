/** Returns the CSRF token from the browser's cookies, or null if not found. */
function getCsrfToken() {
    const name = 'csrftoken';
    for (const cookie of document.cookie.split(';')) {
        const [key, value] = cookie.trim().split('=');
        if (key === name) return decodeURIComponent(value);
    }
    return null;
}

// Global state for ATR results per page
let globalHtrEngineResult = '';
let globalMllmOnlyResult = '';
let globalMllmMergedResult = '';
let globalUserVersion1 = '';
let globalUserVersion2 = '';
let currentImages = [];
let currentPageIndex = 0;
let pageResults = [];
let contentOfFinalTEI = '';

// CodeMirror editor instances
let leftEditor = null;
let rightEditor = null;

// Diff navigation state
let diffMarkers = [];       // One entry per differing line, used for prev/next navigation
let currentDiffIndex = -1;  // Index into diffMarkers of the currently highlighted diff
let currentLineMarkers = []; // Temporary "active" highlight markers, cleared on navigation


// ---------------------------------------------------------------------------
// Diff highlighting
// ---------------------------------------------------------------------------

/**
 * Computes character-level diffs between the left and right editors line by
 * line and marks differences using CodeMirror's markText API.
 *
 * Strategy:
 *   - Lines are compared positionally (line N left vs. line N right).
 *   - diff_match_patch is used for character-level granularity within each
 *     differing line.
 *   - Deletions are marked on the left editor (cm-diff-delete).
 *   - Insertions are marked on the right editor (cm-diff-insert).
 *   - Only the first insertion per line is tracked in diffMarkers so that
 *     prev/next navigation moves one line at a time.
 */
function highlightDifferences() {
    if (!leftEditor || !rightEditor) return;

    // Clear all previous highlights
    leftEditor.getAllMarks().forEach(mark => mark.clear());
    rightEditor.getAllMarks().forEach(mark => mark.clear());
    diffMarkers = [];
    currentLineMarkers = [];

    // Split text by line breaks
    const leftLines = leftEditor.getValue().split('\n');
    const rightLines = rightEditor.getValue().split('\n');
    const maxLines = Math.max(leftLines.length, rightLines.length);

    const dmp = new diff_match_patch();

    // Compare lines
    for (let lineNum = 0; lineNum < maxLines; lineNum++) {
        const leftLine = leftLines[lineNum] ?? '';
        const rightLine = rightLines[lineNum] ?? '';

        if (leftLine === rightLine) continue;

        const diffs = dmp.diff_main(leftLine, rightLine);

        // Improve readability
        dmp.diff_cleanupSemantic(diffs);

        // Only track the first insertion per line for navigation
        let leftPos = 0;
        let rightPos = 0;
        let lineTracked = false;

        for (const [operation, text] of diffs) {
            const len = text.length;

            if (operation === -1) {
                // Text present in left but not in right → mark as deleted on left
                leftEditor.markText(
                    {line: lineNum, ch: leftPos},
                    {line: lineNum, ch: leftPos + len},
                    {className: 'cm-diff-delete'}
                );
                leftPos += len;
            } else if (operation === 1) {
                // Text present in right but not in left → mark as inserted on right
                const from = {line: lineNum, ch: rightPos};
                rightEditor.markText(
                    from,
                    {line: lineNum, ch: rightPos + len},
                    {className: 'cm-diff-insert'}
                );

                if (!lineTracked) {
                    diffMarkers.push({pos: from, lineNumber: lineNum});
                    lineTracked = true;
                }
                rightPos += len;
            } else {
                // Unchanged text (operation === 0)
                leftPos += len;
                rightPos += len;
            }
        }
    }

    updateDiffNavigation();

    // Restore active highlight if the current index is still valid
    if (currentDiffIndex >= 0 && currentDiffIndex < diffMarkers.length) {
        highlightCurrentDiff();
    }
}

/** Enables/disables the diff navigation buttons and updates the counter label. */
function updateDiffNavigation() {
    const prevBtn = document.getElementById('prev-diff-btn');
    const nextBtn = document.getElementById('next-diff-btn');
    const counter = document.getElementById('diff-counter');
    const copyToRightBtn = document.getElementById('copy-to-right-btn');
    const copyToLeftBtn = document.getElementById('copy-to-left-btn');

    const noDiffs = diffMarkers.length === 0;
    const noDiffSelected = currentDiffIndex < 0;

    prevBtn.disabled = noDiffs || currentDiffIndex <= 0;
    nextBtn.disabled = noDiffs || currentDiffIndex >= diffMarkers.length - 1;
    counter.textContent = noDiffs ? '0/0' : `${currentDiffIndex + 1}/${diffMarkers.length}`;

    if (copyToRightBtn) copyToRightBtn.disabled = noDiffs || noDiffSelected;
    if (copyToLeftBtn) copyToLeftBtn.disabled = noDiffs || noDiffSelected;
}

/** Moves to the next diff entry and highlights it. */
function goToNextDiff() {
    if (currentDiffIndex < diffMarkers.length - 1) {
        currentDiffIndex++;
        highlightCurrentDiff();
    }
}

/** Moves to the previous diff entry and highlights it. */
function goToPrevDiff() {
    if (currentDiffIndex > 0) {
        currentDiffIndex--;
        highlightCurrentDiff();
    }
}

/**
 * Adds a yellow "active" highlight (cm-diff-current) to every marked span on
 * the current diff's line in both editors, then scrolls the right editor to
 * bring that line into view.
 *
 * The yellow markers are stored in currentLineMarkers so they can be cleared
 * when the user navigates to a different diff.
 */
function highlightCurrentDiff() {
    currentLineMarkers.forEach(marker => marker.clear());
    currentLineMarkers = [];

    if (currentDiffIndex < 0 || currentDiffIndex >= diffMarkers.length) return;

    const {lineNumber, pos} = diffMarkers[currentDiffIndex];

    // Helper: adds cm-diff-current marks for all existing marks on the target line
    const markLine = (editor) => {
        editor.getAllMarks().forEach(mark => {
            const range = mark.find();
            if (range && range.from.line === lineNumber && range.to.line === lineNumber) {
                currentLineMarkers.push(
                    editor.markText(range.from, range.to, {className: 'cm-diff-current'})
                );
            }
        });
    };

    markLine(leftEditor);
    markLine(rightEditor);

    rightEditor.scrollIntoView(pos, 100);
    updateDiffNavigation();
}

/**
 * Copies the current diff line from one editor to the other, then re-runs
 * diff highlighting and restores the navigation index.
 *
 * A 100 ms timeout is used to let CodeMirror finish its internal change
 * processing before we re-query marks.
 *
 * @param {'left'|'right'} source - The editor whose line content is used as the source.
 */
function copyDiffLine(source) {
    if (currentDiffIndex < 0 || currentDiffIndex >= diffMarkers.length) return;

    const {lineNumber} = diffMarkers[currentDiffIndex];
    const [srcEditor, dstEditor] = source === 'left'
        ? [leftEditor, rightEditor]
        : [rightEditor, leftEditor];

    const sourceLine = srcEditor.getLine(lineNumber);
    if (sourceLine === undefined) return;

    const savedIndex = currentDiffIndex;
    const dstLineLen = dstEditor.getLine(lineNumber)?.length ?? 0;
    dstEditor.replaceRange(sourceLine, {line: lineNumber, ch: 0}, {line: lineNumber, ch: dstLineLen});

    setTimeout(() => {
        highlightDifferences();
        currentDiffIndex = Math.min(savedIndex, diffMarkers.length - 1);
        if (diffMarkers.length > 0) {
            highlightCurrentDiff();
        } else {
            currentDiffIndex = -1;
            updateDiffNavigation();
        }
    }, 100);
}

/** Copies the current diff line from the left editor into the right editor. */
const copyDiffToRight = () => copyDiffLine('left');

/** Copies the current diff line from the right editor into the left editor. */
const copyDiffToLeft = () => copyDiffLine('right');

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/**
 * Returns a debounced version of func that delays invocation by `wait` ms.
 * Repeated calls within the wait window reset the timer.
 * @param {Function} func
 * @param {number} wait - Delay in milliseconds.
 */
function debounce(func, wait) {
    let timeout;
    return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func(...args), wait);
    };
}

const debouncedHighlight = debounce(highlightDifferences, 300);

// ---------------------------------------------------------------------------
// DOM ready
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {

    // --- CodeMirror editors ---------------------------------------------------

    const leftTextarea = document.getElementById('codemirror-left');
    const rightTextarea = document.getElementById('codemirror-right');

    if (leftTextarea && rightTextarea) {
        const editorOptions = {
            lineNumbers: true,
            mode: 'text/plain',
            lineWrapping: true
        };

        leftEditor = CodeMirror.fromTextArea(leftTextarea, editorOptions);
        rightEditor = CodeMirror.fromTextArea(rightTextarea, editorOptions);

        // Synchronise scroll position between both editors so the user always
        // sees the same region in both panes. The guard prevents infinite loops
        // when scrollTo triggers a scroll event on the other editor.
        let syncingScroll = false;

        leftEditor.on('scroll', () => {
            if (syncingScroll) return;
            syncingScroll = true;
            const {left, top} = leftEditor.getScrollInfo();
            rightEditor.scrollTo(left, top);
            syncingScroll = false;
        });

        rightEditor.on('scroll', () => {
            if (syncingScroll) return;
            syncingScroll = true;
            const {left, top} = rightEditor.getScrollInfo();
            leftEditor.scrollTo(left, top);
            syncingScroll = false;
        });

        leftEditor.on('change', debouncedHighlight);
        rightEditor.on('change', debouncedHighlight);

        highlightDifferences();
    }

    // --- Diff navigation buttons ---------------------------------------------

    document.getElementById('prev-diff-btn')?.addEventListener('click', goToPrevDiff);
    document.getElementById('next-diff-btn')?.addEventListener('click', goToNextDiff);
    document.getElementById('copy-to-right-btn')?.addEventListener('click', copyDiffToRight);
    document.getElementById('copy-to-left-btn')?.addEventListener('click', copyDiffToLeft);

    // --- OpenSeadragon image viewer ------------------------------------------

    const dropZone = document.getElementById('drop-zone');
    const noImagePlaceholder = document.getElementById('no-image-placeholder');
    const fileInput = document.getElementById('file-input');

    const viewer = OpenSeadragon({
        id: 'openseadragon-viewer',
        prefixUrl: '',
        animationTime: 0.5,
        blendTime: 0.1,
        constrainDuringPan: true,
        maxZoomPixelRatio: 5,
        minZoomLevel: 0.1,
        visibilityRatio: 0.2,
        zoomPerScroll: 1.2,
        showNavigationControl: false
    });

    /**
     * Creates and appends a styled overlay button to the OpenSeadragon viewer element.
     * @param {string} id
     * @param {string} title - Tooltip text.
     * @param {string} svgHtml - Inner SVG markup.
     * @param {string} right - CSS right offset (e.g. '10px').
     * @param {Function} onClick
     */
    function createViewerButton(id, title, svgHtml, right, onClick) {
        const btn = document.createElement('button');
        btn.id = id;
        btn.className = 'openseadragon-button';
        btn.title = title;
        btn.innerHTML = svgHtml;
        Object.assign(btn.style, {
            position: 'absolute', bottom: '10px', right,
            zIndex: '1000', backgroundColor: 'rgba(255,255,255,0.7)',
            border: 'none', borderRadius: '4px', padding: '8px', cursor: 'pointer'
        });
        btn.addEventListener('click', onClick);
        document.getElementById('openseadragon-viewer').appendChild(btn);
    }

    const folderSvg = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>';
    const fullscreenSvg = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path></svg>';

    createViewerButton('upload-button', 'Bild hochladen', folderSvg, '10px', () => fileInput.click());

    createViewerButton('fullscreen-button', 'Vollbild anzeigen', fullscreenSvg, '50px', () => {
        const el = document.getElementById('openseadragon-viewer');
        if (!el.requestFullscreen) return;
        document.fullscreenElement ? document.exitFullscreen() : el.requestFullscreen();
    });

    // --- Drag & drop ---------------------------------------------------------

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('border-blue-500');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('border-blue-500');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('border-blue-500');
        if (e.dataTransfer.files?.length > 0) handleImageFiles(e.dataTransfer.files);
    });

    // Only open the file picker when no image is loaded yet
    dropZone.addEventListener('click', () => {
        const viewerVisible = document.getElementById('openseadragon-viewer').style.display === 'block';
        const placeholderHidden = noImagePlaceholder.style.display === 'none';
        if (!viewerVisible || !placeholderHidden) fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files?.length > 0) handleImageFiles(fileInput.files);
    });

    // --- Image handling ------------------------------------------------------

    /**
     * Handles one or more image files selected by the user.
     * Initialises page results, clears previous editor content, and loads all images.
     * @param {FileList} files - The files selected via drag-and-drop or the file picker.
     */
    function handleImageFiles(files) {
        currentImages = Array.from(files);
        currentPageIndex = 0;

        pageResults = currentImages.map(() => ({
            imageUrl: '',
            ocrEngineResult: '',
            mllmOnlyResult: '',
            mllmMergedResult: '',
            userVersion1: '',
            userVersion2: ''
        }));

        globalHtrEngineResult = '';
        globalMllmOnlyResult = '';
        globalMllmMergedResult = '';

        leftEditor?.setValue('');
        rightEditor?.setValue('');

        const teiTextArea = document.getElementById('tei-text');
        if (teiTextArea) teiTextArea.value = '';

        loadAllImages().then(() => showPage(0));
    }

    /**
     * Reads all images in currentImages as data URLs and stores them in pageResults.
     * @returns {Promise<void>} Resolves when every image has been loaded.
     */
    function loadAllImages() {
        const promises = currentImages.map((file, index) =>
            new Promise((resolve) => {
                const reader = new FileReader();
                reader.onload = (e) => {
                    pageResults[index].imageUrl = e.target.result;
                    resolve();
                };
                reader.readAsDataURL(file);
            })
        );
        return Promise.all(promises);
    }

    /**
     * Displays the image at the given index in the OpenSeadragon viewer
     * and loads the stored OCR results for that page.
     * @param {number} index - Zero-based page index.
     */
    function showPage(index) {
        if (index < 0 || index >= currentImages.length) return;

        currentPageIndex = index;

        document.getElementById('openseadragon-viewer').style.display = 'block';
        noImagePlaceholder.style.display = 'none';

        viewer.open({type: 'image', url: pageResults[index].imageUrl});

        updatePageNavigation();
        loadPageResults(index);
    }

    /**
     * Shows or hides the page navigation controls and updates the prev/next button state
     * based on the current page index.
     */
    function updatePageNavigation() {
        const pageNav = document.getElementById('page-navigation');
        const prevBtn = document.getElementById('prev-page-btn');
        const nextBtn = document.getElementById('next-page-btn');

        if (currentImages.length > 1) {
            pageNav.style.display = 'flex';
            prevBtn.disabled = currentPageIndex === 0;
            nextBtn.disabled = currentPageIndex === currentImages.length - 1;
        } else {
            pageNav.style.display = 'none';
        }
    }

    /**
     * Loads the stored OCR results for the given page index into the global variables
     * and refreshes the editor panels accordingly.
     * @param {number} index - Zero-based page index.
     */
    function loadPageResults(index) {
        const result = pageResults[index];
        globalHtrEngineResult = result.ocrEngineResult;
        globalMllmOnlyResult = result.mllmOnlyResult;
        globalMllmMergedResult = result.mllmMergedResult;
        globalUserVersion1 = result.userVersion1;
        globalUserVersion2 = result.userVersion2;
        updateDropdownSelections();
    }

    /** Advances to the next page if one exists. */
    function nextPage() {
        if (currentPageIndex < currentImages.length - 1) showPage(currentPageIndex + 1);
    }

    /** Moves to the previous page if one exists. */
    function previousPage() {
        if (currentPageIndex > 0) showPage(currentPageIndex - 1);
    }

    document.getElementById('prev-page-btn')?.addEventListener('click', previousPage);
    document.getElementById('next-page-btn')?.addEventListener('click', nextPage);

    // --- Status window -------------------------------------------------------

    let statusPollInterval = null;

    /**
     * Shows the status overlay with the given message and an animated spinner.
     * @param {string} message - The status message to display.
     */
    function showStatusWindow(message) {
        const statusWindow = document.getElementById('status-window');
        const statusText = document.getElementById('status-text');
        const statusSpinner = document.getElementById('status-spinner');
        if (statusWindow && statusText) {
            statusText.textContent = message;
            statusWindow.classList.remove('hidden');
            statusSpinner?.classList.remove('hidden');
        }
    }

    /** Hides the status overlay and its spinner. */
    function hideStatusWindow() {
        const statusWindow = document.getElementById('status-window');
        const statusSpinner = document.getElementById('status-spinner');
        if (statusWindow) {
            statusWindow.classList.add('hidden');
            statusSpinner?.classList.add('hidden');
        }
    }

    /**
     * Updates the text shown in the status overlay without changing its visibility.
     * @param {string} message
     */
    function updateStatusWindow(message) {
        const statusText = document.getElementById('status-text');
        if (statusText) statusText.textContent = message;
    }

    document.getElementById('status-close-btn')?.addEventListener('click', () => {
        hideStatusWindow();
        clearInterval(statusPollInterval);
        statusPollInterval = null;
    });

    // --- Polling -------------------------------------------------------------

    /** Re-enables the main action buttons after a job finishes or fails. */
    function reEnableActionButtons() {
        document.getElementById('start-atr-btn').disabled = false;
        document.getElementById('create-tei-btn').disabled = false;
    }

    /** Clears the active poll interval and resets the reference to null. */
    function stopPolling() {
        clearInterval(statusPollInterval);
        statusPollInterval = null;
    }

    /**
     * Polls the server once for the status of a background job and updates the UI.
     * Stops polling automatically when the job is complete or an error occurs.
     *
     * On completion the function handles two result shapes:
     *   - data.results  → array of per-page OCR results
     *   - data.result_tei → final TEI-XML string
     *
     * @param {string} processId - The UUID of the background job to check.
     */
    function pollStatus(processId) {
        fetch(`/check_status/?process_id=${processId}`)
            .then(response => response.json())
            .then(data => {
                const status = data.status;
                updateStatusWindow(status);

                if (status === 'Verarbeitung abgeschlossen') {
                    stopPolling();
                    reEnableActionButtons();
                    setTimeout(hideStatusWindow, 2000);

                    // Store per-page OCR results returned by the backend
                    if (Array.isArray(data.results)) {
                        data.results.forEach(pageResult => {
                            const pageIndex = pageResult.page;
                            if (pageIndex >= 0 && pageIndex < pageResults.length) {
                                if (pageResult.mllm_only_result) pageResults[pageIndex].mllmOnlyResult = pageResult.mllm_only_result;
                                if (pageResult.ocr_engine_result) pageResults[pageIndex].ocrEngineResult = pageResult.ocr_engine_result;
                                if (pageResult.mllm_merged_result) pageResults[pageIndex].mllmMergedResult = pageResult.mllm_merged_result;
                            }
                        });
                        loadPageResults(currentPageIndex);
                    }

                    // Store final TEI result
                    if (data.result_tei) {
                        document.getElementById('tei-text').value = data.result_tei;
                        contentOfFinalTEI = data.result_text_content_only;
                    }

                } else if (data.error || status.startsWith('Error')) {
                    stopPolling();
                    reEnableActionButtons();
                    alert(`Fehler: ${status}`);
                    hideStatusWindow();
                }
            })
            .catch(error => {
                console.error('Error checking status:', error);
                stopPolling();
                hideStatusWindow();
            });
    }

    /**
     * Starts a polling interval that calls pollStatus every `intervalMs` ms
     * and also fires immediately on the first call.
     * @param {string} processId
     * @param {number} intervalMs
     */
    function startPolling(processId, intervalMs) {
        statusPollInterval = setInterval(() => pollStatus(processId), intervalMs);
        pollStatus(processId);
    }

    // --- ATR processing ------------------------------------------------------

    document.getElementById('start-atr-btn').addEventListener('click', () => {
        if (currentImages.length === 0) {
            alert('Bitte zuerst ein Bild hochladen.');
            return;
        }
        uploadAndProcessAllImages();
    });

    /**
     * Uploads all currently loaded images to the server and starts the ATR pipeline.
     * Disables action buttons while running and re-enables them when complete or on error.
     */
    function uploadAndProcessAllImages() {
        leftEditor?.setValue('');
        rightEditor?.setValue('');

        globalHtrEngineResult = '';
        globalMllmOnlyResult = '';
        globalMllmMergedResult = '';

        const formData = new FormData();
        currentImages.forEach(file => formData.append('images', file));
        formData.append('transkribus-model', document.getElementById('transkribus-model').value);
        formData.append('multimodal-llm-ocr', document.getElementById('multimodal-llm-ocr').value);
        formData.append('temperature-ocr', document.getElementById('temperature-ocr').value);
        formData.append('mode', document.getElementById('mode').value);

        document.getElementById('start-atr-btn').disabled = true;
        document.getElementById('create-tei-btn').disabled = true;
        showStatusWindow('Texterkennung gestartet – alle Seiten werden verarbeitet...');

        fetch('/upload_image/', {
            method: 'POST',
            headers: {'X-CSRFToken': getCsrfToken()},
            body: formData
        })
            .then(response => {
                if (!response.ok) throw new Error(`HTTP-Fehler! Status: ${response.status}`);
                return response.json();
            })
            .then(data => {
                if (!data.process_id) throw new Error('Keine Prozess-ID erhalten');
                startPolling(data.process_id, 5000);
            })
            .catch(error => {
                console.error('Fehler beim Hochladen des Bildes:', error);
                reEnableActionButtons();
                alert('Verarbeitungsfehler: ' + error.message);
                hideStatusWindow();
            });
    }

    // --- TEI generation ------------------------------------------------------

    document.getElementById('create-tei-btn').addEventListener('click', () => {
        // Collapse the comparison section when TEI generation starts
        const comparisonSection = document.getElementById('comparison-section');
        const toggleBtn = document.getElementById('toggle-comparison-btn');
        comparisonSection?.classList.add('hidden-smooth');
        if (toggleBtn) toggleBtn.textContent = 'Vergleichsansicht einblenden';

        if (currentImages.length > 1) {
            createMultiPageTEI();
        } else {
            createSinglePageTEI();
        }
    });

    /**
     * Builds a FormData object with the shared TEI generation parameters
     * (LLM selection, prompt type, custom prompt, temperature, mode).
     * @returns {FormData}
     */
    function buildTeiFormData() {
        const promptType = document.getElementById('prompt-transformation-tei').value;
        const customPromptText = promptType === 'prompt-tei-custom'
            ? document.getElementById('custom-prompt-text').value
            : '';

        const formData = new FormData();
        formData.append('multimodal-llm-tei', document.getElementById('multimodal-llm-tei').value);
        formData.append('prompt-transformation-tei', promptType);
        formData.append('custom-prompt-text', customPromptText);
        formData.append('temperature-tei', document.getElementById('temperature-tei').value);
        formData.append('mode', document.getElementById('mode').value);
        return formData;
    }

    /**
     * Sends the formData to /create_tei/ and starts polling for the result.
     * Shared by single-page and multi-page TEI generation.
     * @param {FormData} formData
     */
    function submitTeiRequest(formData) {
        document.getElementById('start-atr-btn').disabled = true;
        document.getElementById('create-tei-btn').disabled = true;
        showStatusWindow('TEI-XML-Erstellung gestartet');

        fetch('/create_tei/', {
            method: 'POST',
            headers: {'X-CSRFToken': getCsrfToken()},
            body: formData
        })
            .then(response => {
                if (!response.ok) throw new Error(`HTTP-Fehler! Status: ${response.status}`);
                return response.json();
            })
            .then(data => {
                if (!data.process_id) throw new Error('Keine Prozess-ID erhalten');
                startPolling(data.process_id, 3000);
            })
            .catch(error => {
                console.error('Fehler bei der TEI-XML-Erstellung:', error);
                reEnableActionButtons();
                document.getElementById('tei-text').value = 'Fehler bei der TEI-XML-Erstellung. Bitte erneut versuchen.';
                hideStatusWindow();
            });
    }

    /**
     * Sends the current right-editor text and the active image to the server
     * to generate TEI-XML for a single-page document.
     */
    function createSinglePageTEI() {
        const mergedText = rightEditor ? rightEditor.getValue() : '';
        const image = currentImages[currentPageIndex] ?? null;

        if (!mergedText) {
            alert('Die rechte Vergleichsansicht ist leer. Bitte führen Sie zuerst die Texterkennung durch.');
            return;
        }
        if (!image) {
            console.error('Bild ist nicht definiert oder leer.');
            alert('Bitte laden zuerst ein Bild hoch.');
            return;
        }

        const combinedTextArea = document.getElementById('combined-text');
        if (combinedTextArea) combinedTextArea.value = mergedText;

        const formData = buildTeiFormData();
        formData.append('image', image);
        formData.append('merged_text', mergedText);

        submitTeiRequest(formData);
    }

    /**
     * Collects the selected text version from each page, concatenates them, and sends
     * all images and the combined text to the server to generate TEI-XML for a
     * multi-page document.
     */
    function createMultiPageTEI() {
        const selectedVersion = document.getElementById('version-for-tei').value;

        // Map dropdown value to the corresponding pageResult property
        const versionKey = {
            'transkribus': 'ocrEngineResult',
            'multimodal-model': 'mllmOnlyResult',
            'merged': 'mllmMergedResult',
            'user-version-1': 'userVersion1',
            'user-version-2': 'userVersion2'
        };

        const formData = buildTeiFormData();
        let combinedText = '';

        for (let i = 0; i < currentImages.length; i++) {
            const key = versionKey[selectedVersion] ?? 'mllmMergedResult';
            const selectedText = pageResults[i][key];

            if (!selectedText) {
                alert(`Für Seite ${i + 1} liegt kein Texterkennungsergebnis vor. Bitte führen zuerst die Texterkennung durch.`);
                return;
            }

            combinedText += (i > 0 ? '\n\n' : '') + selectedText;
            formData.append(`image_${i}`, currentImages[i]);
        }

        const combinedTextArea = document.getElementById('combined-text');
        if (combinedTextArea) combinedTextArea.value = combinedText;

        formData.append('merged_text', combinedText);
        formData.append('num_pages', currentImages.length.toString());

        submitTeiRequest(formData);
    }

    // --- Comparison view toggle ----------------------------------------------

    document.getElementById('toggle-comparison-btn')?.addEventListener('click', (e) => {
        e.preventDefault();
        const comparisonSection = document.getElementById('comparison-section');
        const toggleBtn = document.getElementById('toggle-comparison-btn');
        if (!comparisonSection) return;

        const isHidden = comparisonSection.classList.contains('hidden-smooth');
        comparisonSection.classList.toggle('hidden-smooth', !isHidden);
        toggleBtn.textContent = isHidden ? 'Vergleichsansicht ausblenden' : 'Vergleichsansicht einblenden';
    });

    // --- Dropdown result selectors -------------------------------------------

    const dropdownCompareLeft = document.getElementById('dropdown-compare-left');
    const dropdownCompareRight = document.getElementById('dropdown-compare-right');

    /**
     * Persists the current editor content into the appropriate user-version slot
     * (user-version-1 or user-version-2) for the active page, based on the
     * currently selected dropdown values.
     *
     * Called on a 1-second interval to auto-save edits made by the user.
     */
    function saveUserVersion() {
        const lhsContent = leftEditor?.getValue() ?? '';
        const rhsContent = rightEditor?.getValue() ?? '';

        if (dropdownCompareLeft.value === 'user-version-1' && globalUserVersion1 !== lhsContent) {
            globalUserVersion1 = lhsContent;
            pageResults[currentPageIndex].userVersion1 = lhsContent;
        }
        if (dropdownCompareRight.value === 'user-version-1' && globalUserVersion1 !== rhsContent) {
            globalUserVersion1 = rhsContent;
            pageResults[currentPageIndex].userVersion1 = rhsContent;
        }
        if (dropdownCompareLeft.value === 'user-version-2' && globalUserVersion2 !== lhsContent) {
            globalUserVersion2 = lhsContent;
            pageResults[currentPageIndex].userVersion2 = lhsContent;
        }
        if (dropdownCompareRight.value === 'user-version-2' && globalUserVersion2 !== rhsContent) {
            globalUserVersion2 = rhsContent;
            pageResults[currentPageIndex].userVersion2 = rhsContent;
        }
    }

    // Auto-save user edits every second when a user-version slot is active
    setInterval(() => {
        const userVersionActive =
            dropdownCompareLeft.value.startsWith('user-version') ||
            dropdownCompareRight.value.startsWith('user-version');
        if (userVersionActive) saveUserVersion();
    }, 1000);

    /** Refreshes both editor panels to reflect the currently selected dropdown values. */
    function updateDropdownSelections() {
        updateDropdownSide(dropdownCompareLeft.value, 'left');
        updateDropdownSide(dropdownCompareRight.value, 'right');
    }

    /**
     * Loads the text content for the given result type into the left or right editor.
     * After setting the content, diff highlighting runs automatically and the first
     * difference (if any) is selected.
     *
     * @param {'transkribus'|'multimodal-model'|'merged'|'user-version-1'|'user-version-2'} value
     * @param {'left'|'right'} side - Which editor to update.
     */
    function updateDropdownSide(value, side) {
        const contentMap = {
            'transkribus': globalHtrEngineResult,
            'multimodal-model': globalMllmOnlyResult,
            'merged': globalMllmMergedResult,
            'user-version-1': globalUserVersion1,
            'user-version-2': globalUserVersion2
        };

        const editor = side === 'left' ? leftEditor : rightEditor;
        if (!editor) return;

        editor.setValue(contentMap[value] ?? '');

        // Re-compute diffs after the editor has finished updating
        setTimeout(() => {
            highlightDifferences();
            if (diffMarkers.length > 0) {
                currentDiffIndex = 0;
                highlightCurrentDiff();
            } else {
                currentDiffIndex = -1;
            }
        }, 100);
    }

    dropdownCompareLeft.addEventListener('change', () => updateDropdownSide(dropdownCompareLeft.value, 'left'));
    dropdownCompareRight.addEventListener('change', () => updateDropdownSide(dropdownCompareRight.value, 'right'));

    // --- Clipboard -----------------------------------------------------------

    /**
     * Copies the given text to the clipboard.
     * Uses the modern Clipboard API when available (requires a secure context),
     * falling back to a legacy execCommand approach for older browsers.
     * @param {string} text - The text to copy.
     */
    function copyToClipboard(text) {
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(text).catch(() => fallbackCopyTextToClipboard(text));
        } else {
            fallbackCopyTextToClipboard(text);
        }
    }

    /**
     * Legacy clipboard copy using a temporary off-screen textarea and execCommand.
     * Used when the Clipboard API is unavailable (non-HTTPS or older browsers).
     * @param {string} text - The text to copy.
     */
    function fallbackCopyTextToClipboard(text) {
        try {
            const textArea = document.createElement('textarea');
            textArea.value = text;
            Object.assign(textArea.style, {position: 'fixed', left: '-999999px', top: '-999999px'});
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            try {
                document.execCommand('copy');
            } catch (execErr) {
                console.warn('execCommand is not supported', execErr);
            }
            document.body.removeChild(textArea);
        } catch (err) {
            console.error('Fallback: copy failed', err);
        }
    }

    document.getElementById('copy-left-btn').addEventListener('click', () => copyToClipboard(leftEditor?.getValue() ?? ''));
    document.getElementById('copy-right-btn').addEventListener('click', () => copyToClipboard(rightEditor?.getValue() ?? ''));
    document.getElementById('copy-result-btn').addEventListener('click', () => copyToClipboard(document.getElementById('tei-text').value));

    // --- Content check -------------------------------------------------------

    /**
     * Compares the plain-text content of the generated TEI with the right editor's
     * current content and alerts the user with a summary of differences.
     *
     * Normalises line endings before comparison to avoid false positives from
     * CRLF vs LF differences.
     */
    document.getElementById('check-content-btn').addEventListener('click', () => {
        const compareRightContent = rightEditor?.getValue() ?? '';

        if (!contentOfFinalTEI || !compareRightContent) {
            alert('Es gibt keinen Inhalt zum Vergleichen. Bitte zuerst das TEI-XML erstellen.');
            return;
        }

        const normalize = str => str.trim().replace(/\r\n|\r/g, '\n');
        const normalizedTEI = normalize(contentOfFinalTEI);
        const normalizedCompare = normalize(compareRightContent);

        if (normalizedTEI === normalizedCompare) {
            alert('Inhalt stimmt überein. Das Endergebnis enthält denselben Text wie die rechte Vergleichsansicht.');
            return;
        }

        const teiLines = normalizedTEI.split('\n');
        const compareLines = normalizedCompare.split('\n');
        const maxLength = Math.max(teiLines.length, compareLines.length);
        const differences = [];

        for (let i = 0; i < maxLength; i++) {
            const teiLine = teiLines[i]?.trim() ?? null;
            const compareLine = compareLines[i]?.trim() ?? null;

            if (teiLine === compareLine) continue;

            if (teiLine === null) {
                differences.push(`Zeile ${i + 1}: Fehlt im Ausgabetext: „${compareLine}"`);
            } else if (compareLine === null) {
                differences.push(`Zeile ${i + 1}: Fehlt im Eingabetext: „${teiLine}"`);
            } else {
                differences.push(`Zeile ${i + 1}: Unterschied – TEI: „${teiLine}" vs. Vergleich: „${compareLine}"`);
            }
        }

        const maxDifferences = 10;
        let message = 'Folgende Unterschiede wurden gefunden:\n\n';
        message += differences.slice(0, maxDifferences).join('\n');
        if (differences.length > maxDifferences) {
            message += `\n\n... und ${differences.length - maxDifferences} weitere Unterschiede.`;
        }

        alert(message);
    });

});