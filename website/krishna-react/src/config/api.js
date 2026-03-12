// API Configuration
// This file centralizes all API endpoint URLs

const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://talk-to-krishna-japan.onrender.com';

export const API_ENDPOINTS = {
    // Authentication
    LOGIN: `${API_BASE_URL}/api/login`,
    SIGNUP: `${API_BASE_URL}/api/signup`,
    FORGOT_PASSWORD: `${API_BASE_URL}/api/forgot-password`,
    RESET_PASSWORD: `${API_BASE_URL}/api/reset-password`,

    // AI Chat
    ASK: `${API_BASE_URL}/api/ask`,
    HISTORY: `${API_BASE_URL}/api/history`,

    // Audio (if needed)
    SPEAK: `${API_BASE_URL}/api/speak`,
    TRANSCRIBE: `${API_BASE_URL}/api/transcribe`,

    // Payment & Access
    GRANT_ACCESS: `${API_BASE_URL}/api/grant-access`,
    VALIDATE_COUPON: `${API_BASE_URL}/api/validate-coupon`,
};

export default API_BASE_URL;
