const mongoose = require('mongoose');

const connectDB = async () => {
    try {
        const mongoURI = process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/land_acquisition_db';
        const conn = await mongoose.connect(mongoURI, {
            serverSelectionTimeoutMS: 3000
        });
        console.log(`MongoDB Connected: ${conn.connection.host}/${conn.connection.name}`);
    } catch (error) {
        console.warn('MongoDB Warning:', error.message);
        console.warn('Using in-memory storage fallback if MongoDB is not reachable.');
    }
};

module.exports = connectDB;
