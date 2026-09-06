const express = require('express');
const path = require('path');
const engine = require('ejs-mate');
const connectDB = require('./config/db');
const caseRoutes = require('./routes/caseRoutes');

const app = express();
const PORT = process.env.PORT || 3000;

// Connect to MongoDB
connectDB();

// Template Engine
app.engine('ejs', engine);
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Middleware
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Routes
app.use('/', caseRoutes);

// 404 Handler
app.use((req, res) => {
    res.status(404).render('listing/Home', {
        title: 'Page Not Found',
        stats: { totalCases: 0, highRisk: 0, mediumRisk: 0, lowRisk: 0 }
    });
});

app.listen(PORT, () => {
    console.log(`Express Backend Server running on http://localhost:${PORT}`);
    console.log(`Frontend presentation layer ready (EJS + Bootstrap)`);
    console.log(`FastAPI connection configured for http://localhost:8000`);
});

module.exports = app;
