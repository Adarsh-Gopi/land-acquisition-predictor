const mongoose = require('mongoose');

const CaseSchema = new mongoose.Schema({
    case_id: { type: String, required: true, trim: true },
    project_name: { type: String, default: 'Infrastructure Corridor Project' },
    state: { type: String, default: 'Uttar Pradesh' },
    district_type: { type: String, default: 'Rural' },
    project_type: { type: String, default: 'Railway' },
    land_type: { type: String, default: 'Agricultural' },
    notification_stage: { type: String, default: '3A issued' },
    land_area_hectares: { type: Number, default: 0 },
    land_area_acres: { type: Number, default: 0 },
    affected_owner_count: { type: Number, default: 0 },
    is_multi_village: { type: Number, default: 0 },
    has_title_dispute: { type: Number, default: 0 },
    has_court_case: { type: Number, default: 0 },
    has_objection: { type: Number, default: 0 },
    compensation_estimate_inr_lakh: { type: Number, default: 0 },
    compensation_funding_available: { type: Number, default: 1 },
    days_since_case_opened: { type: Number, default: 0 },
    land_notified_percent: { type: Number, default: 0 },
    award_completed_percent: { type: Number, default: 0 },
    compensation_disbursed_percent: { type: Number, default: 0 },
    possession_completed_percent: { type: Number, default: 0 },
    
    // ML Prediction Output
    delay_probability: { type: Number, default: 0 },
    delay_probability_percent: { type: String, default: '0%' },
    risk_level: { type: String, enum: ['Low', 'Medium', 'High'], default: 'Medium' },
    is_delayed_predicted: { type: Number, default: 0 },
    risk_distribution: {
        low: { type: Number, default: 0 },
        medium: { type: Number, default: 0 },
        high: { type: Number, default: 0 }
    },
    top_risk_factors: [{ type: String }],
    recommended_preventive_actions: [{ type: String }],
    processing_time_ms: { type: Number, default: 0 }
}, { timestamps: true });

module.exports = mongoose.model('Case', CaseSchema);
