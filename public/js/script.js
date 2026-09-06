// Client-side interactions for Land Acquisition UI

document.addEventListener('DOMContentLoaded', () => {
    const areaAcres = document.getElementById('areaAcres');
    const areaHectares = document.getElementById('areaHectares');

    // Auto-sync between Acres and Hectares
    if (areaAcres && areaHectares) {
        areaAcres.addEventListener('input', () => {
            const val = parseFloat(areaAcres.value);
            if (!isNaN(val)) {
                areaHectares.value = (val * 0.404686).toFixed(2);
            }
        });

        areaHectares.addEventListener('input', () => {
            const val = parseFloat(areaHectares.value);
            if (!isNaN(val)) {
                areaAcres.value = (val * 2.47105).toFixed(2);
            }
        });
    }
});
