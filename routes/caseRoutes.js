const express = require('express');
const router = express.Router();
const Case = require('../database/Case');
const { predictDelay } = require('../services/mlService');

// In-memory fallback if MongoDB is not yet running
let memoryCases = [];

// 1. Home Page (GET /)
router.get('/', async (req, res) => {
    try {
        let totalCases = 0;
        let highRisk = 0;
        let mediumRisk = 0;
        let lowRisk = 0;

        try {
            totalCases = await Case.countDocuments();
            highRisk = await Case.countDocuments({ risk_level: 'High' });
            mediumRisk = await Case.countDocuments({ risk_level: 'Medium' });
            lowRisk = await Case.countDocuments({ risk_level: 'Low' });
        } catch (dbErr) {
            totalCases = memoryCases.length;
            highRisk = memoryCases.filter(c => c.risk_level === 'High').length;
            mediumRisk = memoryCases.filter(c => c.risk_level === 'Medium').length;
            lowRisk = memoryCases.filter(c => c.risk_level === 'Low').length;
        }

        res.render('listing/Home', {
            title: 'Early Detection of Land Acquisition Delays',
            stats: { totalCases, highRisk, mediumRisk, lowRisk }
        });
    } catch (err) {
        console.error(err);
        res.render('listing/Home', { title: 'Early Detection of Land Acquisition Delays', stats: { totalCases: 0, highRisk: 0, mediumRisk: 0, lowRisk: 0 } });
    }
});

// 2. Input Form (GET /predict or GET /listing/main)
router.get('/predict', (req, res) => {
    res.render('listing/main', { title: 'New Land Acquisition Case Assessment' });
});

router.get('/listing/main', (req, res) => {
    res.redirect('/predict');
});

// 3. Process Prediction (POST /predict)
router.post('/predict', async (req, res) => {
    try {
        const body = req.body;
        
        const acres = parseFloat(body.land_area_acres) || 0;
        let hectares = parseFloat(body.land_area_hectares) || 0;
        if (!hectares && acres) {
            hectares = Math.round(acres * 0.404686 * 100) / 100;
        }

        const caseData = {
            case_id: body.case_id || `LA-${Date.now().toString().slice(-6)}`,
            project_name: body.project_name || 'Infrastructure Project',
            state: body.state || 'Uttar Pradesh',
            district_type: body.district_type || 'Rural',
            project_type: body.project_type || 'Railway',
            land_type: body.land_type || 'Agricultural',
            notification_stage: body.notification_stage || '3A issued',
            land_area_hectares: hectares || 2.5,
            land_area_acres: acres || Math.round(hectares * 2.47105 * 100) / 100,
            affected_owner_count: parseInt(body.affected_owner_count) || 10,
            is_multi_village: body.is_multi_village === '1' || body.is_multi_village === 1 ? 1 : 0,
            has_title_dispute: body.has_title_dispute === '1' || body.has_title_dispute === 1 ? 1 : 0,
            has_court_case: body.has_court_case === '1' || body.has_court_case === 1 ? 1 : 0,
            has_objection: body.has_objection === '1' || body.has_objection === 1 ? 1 : 0,
            compensation_estimate_inr_lakh: parseFloat(body.compensation_estimate_inr_lakh) || 100,
            compensation_funding_available: body.compensation_funding_available === '1' || body.compensation_funding_available === 1 ? 1 : 0,
            days_since_case_opened: parseInt(body.days_since_case_opened) || 120,
            land_notified_percent: parseFloat(body.land_notified_percent) || 50,
            award_completed_percent: parseFloat(body.award_completed_percent) || 30,
            compensation_disbursed_percent: parseFloat(body.compensation_disbursed_percent) || 20,
            possession_completed_percent: parseFloat(body.possession_completed_percent) || 10,
        };

        // Call FastAPI ML API
        const mlResult = await predictDelay(caseData);
        const pred = mlResult.data;

        const fullCase = {
            ...caseData,
            delay_probability: pred.delay_probability,
            delay_probability_percent: pred.delay_probability_percent,
            risk_level: pred.risk_level,
            is_delayed_predicted: pred.is_delayed_predicted,
            risk_distribution: pred.risk_distribution || { low: 0, medium: 0, high: 0 },
            top_risk_factors: pred.top_risk_factors || [],
            recommended_preventive_actions: pred.recommended_preventive_actions || [],
            processing_time_ms: pred.processing_time_ms || 0,
            warning: mlResult.warning || null
        };

        // Save into MongoDB or in-memory
        try {
            const savedRecord = await Case.create(fullCase);
            fullCase._id = savedRecord._id;
        } catch (dbErr) {
            console.warn('Saved to in-memory history:', dbErr.message);
            fullCase._id = `mem-${Date.now()}`;
            memoryCases.unshift(fullCase);
        }

        res.render('listing/show', {
            title: `Risk Analysis: ${fullCase.case_id}`,
            caseItem: fullCase
        });
    } catch (err) {
        console.error('Error handling prediction:', err);
        res.status(500).send('Error analyzing project case.');
    }
});

// 4. View History (GET /history)
router.get('/history', async (req, res) => {
    try {
        let cases = [];
        try {
            cases = await Case.find().sort({ createdAt: -1 });
        } catch (dbErr) {
            cases = memoryCases;
        }

        res.render('listing/history', {
            title: 'Historical Case Assessments',
            cases: cases
        });
    } catch (err) {
        console.error(err);
        res.render('listing/history', { title: 'Historical Case Assessments', cases: [] });
    }
});

// 5. View Single Case Detail (GET /history/:id)
router.get('/history/:id', async (req, res) => {
    try {
        let caseItem = null;
        try {
            caseItem = await Case.findById(req.params.id);
        } catch (dbErr) {
            caseItem = memoryCases.find(c => c._id == req.params.id);
        }

        if (!caseItem) {
            return res.redirect('/history');
        }

        res.render('listing/show', {
            title: `Assessment: ${caseItem.case_id}`,
            caseItem: caseItem
        });
    } catch (err) {
        console.error(err);
        res.redirect('/history');
    }
});

// 6. Delete Case (POST /history/:id/delete)
router.post('/history/:id/delete', async (req, res) => {
    try {
        try {
            await Case.findByIdAndDelete(req.params.id);
        } catch (dbErr) {
            memoryCases = memoryCases.filter(c => c._id != req.params.id);
        }
        res.redirect('/history');
    } catch (err) {
        console.error(err);
        res.redirect('/history');
    }
});


// Direct view for latest or sample assessment (GET /show)
router.get('/show', async (req, res) => {
    try {
        let latest = null;
        try {
            latest = await Case.findOne().sort({ createdAt: -1 });
        } catch (e) {
            latest = memoryCases[0];
        }
        res.render('listing/show', {
            title: latest ? `Latest Assessment: ${latest.case_id}` : 'Sample Assessment',
            caseItem: latest
        });
    } catch (err) {
        res.render('listing/show', { title: 'Assessment', caseItem: null });
    }
});

module.exports = router;
