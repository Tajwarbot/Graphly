/**
 * Security Utilities Module
 * 
 * OWASP-compliant security utilities for Graphly
 * - Rate limiting (token bucket algorithm)
 * - Input validation & sanitization
 * - Environment variable handling
 * 
 * @module security
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

/**
 * Security configuration constants
 * OWASP: Use secure defaults, fail-safe
 */
const SECURITY_CONFIG = {
    // Rate limiting settings
    RATE_LIMIT: {
        API_CALLS_PER_MINUTE: 5,        // Max API calls per minute
        BUCKET_REFILL_RATE_MS: 12000,   // Refill one token every 12 seconds
        STORAGE_KEY: 'graphly_rate_limit'
    },

    // Input validation limits
    INPUT_LIMITS: {
        GRAPH_TITLE_MAX: 100,
        DATASET_NAME_MAX: 50,
        AXIS_LABEL_MAX: 50,
        CSV_MAX_ROWS: 10000,
        CSV_MAX_COLUMNS: 100,
        IMAGE_MAX_SIZE_MB: 20,           // Updated per user request
        NUMERIC_MIN: -1e15,
        NUMERIC_MAX: 1e15
    },

    // Allowed MIME types for image upload
    ALLOWED_IMAGE_TYPES: [
        'image/jpeg',
        'image/png',
        'image/gif',
        'image/webp',
        'image/bmp'
    ]
};

// =============================================================================
// RATE LIMITER
// =============================================================================

/**
 * Token Bucket Rate Limiter
 * Implements client-side rate limiting for API calls
 * 
 * OWASP: Protect against denial of service and API abuse
 */
class RateLimiter {
    constructor(maxTokens = SECURITY_CONFIG.RATE_LIMIT.API_CALLS_PER_MINUTE) {
        this.maxTokens = maxTokens;
        this.refillRateMs = SECURITY_CONFIG.RATE_LIMIT.BUCKET_REFILL_RATE_MS;
        this.storageKey = SECURITY_CONFIG.RATE_LIMIT.STORAGE_KEY;
        this._loadState();
    }

    /**
     * Load rate limit state from localStorage
     * @private
     */
    _loadState() {
        try {
            const stored = localStorage.getItem(this.storageKey);
            if (stored) {
                const state = JSON.parse(stored);
                this.tokens = state.tokens;
                this.lastRefill = state.lastRefill;
            } else {
                this._resetState();
            }
        } catch (e) {
            // OWASP: Fail secure - if localStorage fails, start fresh
            console.warn('[Security] Rate limiter state load failed, resetting');
            this._resetState();
        }
    }

    /**
     * Reset rate limiter to full tokens
     * @private
     */
    _resetState() {
        this.tokens = this.maxTokens;
        this.lastRefill = Date.now();
        this._saveState();
    }

    /**
     * Save rate limit state to localStorage
     * @private
     */
    _saveState() {
        try {
            localStorage.setItem(this.storageKey, JSON.stringify({
                tokens: this.tokens,
                lastRefill: this.lastRefill
            }));
        } catch (e) {
            // Silent fail - rate limiting still works in-memory
        }
    }

    /**
     * Refill tokens based on elapsed time
     * @private
     */
    _refillTokens() {
        const now = Date.now();
        const elapsed = now - this.lastRefill;
        const tokensToAdd = Math.floor(elapsed / this.refillRateMs);

        if (tokensToAdd > 0) {
            this.tokens = Math.min(this.maxTokens, this.tokens + tokensToAdd);
            this.lastRefill = now;
            this._saveState();
        }
    }

    /**
     * Attempt to consume a token for an API call
     * @returns {Object} { allowed: boolean, retryAfterMs?: number, message?: string }
     */
    tryConsume() {
        this._refillTokens();

        if (this.tokens > 0) {
            this.tokens--;
            this._saveState();
            return { allowed: true };
        }

        // Calculate retry-after time
        const retryAfterMs = this.refillRateMs - (Date.now() - this.lastRefill);
        const retryAfterSec = Math.ceil(retryAfterMs / 1000);

        return {
            allowed: false,
            retryAfterMs,
            message: `Rate limit exceeded. Please wait ${retryAfterSec} seconds before trying again.`
        };
    }

    /**
     * Get remaining tokens (for UI display)
     * @returns {number}
     */
    getRemainingTokens() {
        this._refillTokens();
        return this.tokens;
    }

    /**
     * Reset rate limiter (for testing/admin)
     */
    reset() {
        this._resetState();
    }
}

// Singleton instance for the app
export const rateLimiter = new RateLimiter();

// =============================================================================
// INPUT VALIDATION & SANITIZATION
// =============================================================================

/**
 * Validation result object
 * @typedef {Object} ValidationResult
 * @property {boolean} valid - Whether validation passed
 * @property {*} value - Sanitized value (if valid)
 * @property {string} [error] - Error message (if invalid)
 */

/**
 * Input Validator with schema-based validation
 * 
 * OWASP: Validate all input, use allowlists, sanitize output
 */
const InputValidator = {
    /**
     * Sanitize text input - remove HTML tags and dangerous characters
     * OWASP: Encode/escape all output
     * @param {string} input 
     * @returns {string}
     */
    sanitizeText(input) {
        if (typeof input !== 'string') return '';

        return input
            // Remove HTML tags (XSS prevention)
            .replace(/<[^>]*>/g, '')
            // Remove script-like patterns
            .replace(/javascript:/gi, '')
            .replace(/on\w+=/gi, '')
            // Normalize whitespace
            .replace(/\s+/g, ' ')
            .trim();
    },

    /**
     * Validate and sanitize graph title
     * @param {string} title 
     * @returns {ValidationResult}
     */
    validateGraphTitle(title) {
        if (typeof title !== 'string') {
            return { valid: false, error: 'Graph title must be a string' };
        }

        const sanitized = this.sanitizeText(title);

        if (sanitized.length === 0) {
            return { valid: false, error: 'Graph title cannot be empty' };
        }

        if (sanitized.length > SECURITY_CONFIG.INPUT_LIMITS.GRAPH_TITLE_MAX) {
            return {
                valid: false,
                error: `Graph title cannot exceed ${SECURITY_CONFIG.INPUT_LIMITS.GRAPH_TITLE_MAX} characters`
            };
        }

        return { valid: true, value: sanitized };
    },

    /**
     * Validate and sanitize dataset name
     * @param {string} name 
     * @returns {ValidationResult}
     */
    validateDatasetName(name) {
        if (typeof name !== 'string') {
            return { valid: false, error: 'Dataset name must be a string' };
        }

        const sanitized = this.sanitizeText(name);

        if (sanitized.length === 0) {
            return { valid: false, error: 'Dataset name cannot be empty' };
        }

        if (sanitized.length > SECURITY_CONFIG.INPUT_LIMITS.DATASET_NAME_MAX) {
            return {
                valid: false,
                error: `Dataset name cannot exceed ${SECURITY_CONFIG.INPUT_LIMITS.DATASET_NAME_MAX} characters`
            };
        }

        return { valid: true, value: sanitized };
    },

    /**
     * Validate and sanitize axis label
     * @param {string} label 
     * @returns {ValidationResult}
     */
    validateAxisLabel(label) {
        if (typeof label !== 'string') {
            return { valid: false, error: 'Axis label must be a string' };
        }

        const sanitized = this.sanitizeText(label);

        if (sanitized.length > SECURITY_CONFIG.INPUT_LIMITS.AXIS_LABEL_MAX) {
            return {
                valid: false,
                error: `Axis label cannot exceed ${SECURITY_CONFIG.INPUT_LIMITS.AXIS_LABEL_MAX} characters`
            };
        }

        return { valid: true, value: sanitized };
    },

    /**
     * Validate a numeric value
     * @param {*} value 
     * @returns {ValidationResult}
     */
    validateNumeric(value) {
        const num = Number(value);

        if (isNaN(num)) {
            return { valid: false, error: 'Value must be a number' };
        }

        if (!isFinite(num)) {
            return { valid: false, error: 'Value must be finite' };
        }

        if (num < SECURITY_CONFIG.INPUT_LIMITS.NUMERIC_MIN ||
            num > SECURITY_CONFIG.INPUT_LIMITS.NUMERIC_MAX) {
            return { valid: false, error: 'Value is out of acceptable range' };
        }

        return { valid: true, value: num };
    },

    /**
     * Validate CSV/tabular data
     * @param {Array} data 
     * @returns {ValidationResult}
     */
    validateDataArray(data) {
        if (!Array.isArray(data)) {
            return { valid: false, error: 'Data must be an array' };
        }

        if (data.length === 0) {
            return { valid: false, error: 'Data array cannot be empty' };
        }

        if (data.length > SECURITY_CONFIG.INPUT_LIMITS.CSV_MAX_ROWS) {
            return {
                valid: false,
                error: `Data cannot exceed ${SECURITY_CONFIG.INPUT_LIMITS.CSV_MAX_ROWS} rows`
            };
        }

        // Validate first row to check structure
        const firstRow = data[0];
        if (typeof firstRow !== 'object' || firstRow === null) {
            return { valid: false, error: 'Each data row must be an object' };
        }

        const keys = Object.keys(firstRow);
        if (keys.length > SECURITY_CONFIG.INPUT_LIMITS.CSV_MAX_COLUMNS) {
            return {
                valid: false,
                error: `Data cannot exceed ${SECURITY_CONFIG.INPUT_LIMITS.CSV_MAX_COLUMNS} columns`
            };
        }

        // Sanitize all values
        const sanitizedData = data.map((row, rowIndex) => {
            if (typeof row !== 'object' || row === null) {
                throw new Error(`Invalid row at index ${rowIndex}`);
            }

            const sanitizedRow = {};
            for (const [key, value] of Object.entries(row)) {
                // Sanitize key
                const sanitizedKey = this.sanitizeText(String(key)).slice(0, 50);

                // Validate and sanitize value
                if (typeof value === 'number') {
                    const numResult = this.validateNumeric(value);
                    sanitizedRow[sanitizedKey] = numResult.valid ? numResult.value : 0;
                } else if (typeof value === 'string') {
                    // Try to parse as number first
                    const num = parseFloat(value);
                    if (!isNaN(num)) {
                        const numResult = this.validateNumeric(num);
                        sanitizedRow[sanitizedKey] = numResult.valid ? numResult.value : 0;
                    } else {
                        sanitizedRow[sanitizedKey] = this.sanitizeText(value).slice(0, 100);
                    }
                } else {
                    sanitizedRow[sanitizedKey] = value;
                }
            }
            return sanitizedRow;
        });

        return { valid: true, value: sanitizedData };
    },

    /**
     * Validate image file for upload
     * @param {File} file 
     * @returns {ValidationResult}
     */
    validateImageFile(file) {
        if (!(file instanceof File)) {
            return { valid: false, error: 'Invalid file object' };
        }

        // Check MIME type (OWASP: Validate file type)
        if (!SECURITY_CONFIG.ALLOWED_IMAGE_TYPES.includes(file.type)) {
            return {
                valid: false,
                error: `Invalid file type. Allowed: ${SECURITY_CONFIG.ALLOWED_IMAGE_TYPES.join(', ')}`
            };
        }

        // Check file size
        const maxSizeBytes = SECURITY_CONFIG.INPUT_LIMITS.IMAGE_MAX_SIZE_MB * 1024 * 1024;
        if (file.size > maxSizeBytes) {
            return {
                valid: false,
                error: `File size cannot exceed ${SECURITY_CONFIG.INPUT_LIMITS.IMAGE_MAX_SIZE_MB}MB`
            };
        }

        return { valid: true, value: file };
    }
};

export { InputValidator, SECURITY_CONFIG };

// =============================================================================
// ENVIRONMENT & API KEY HANDLING
// =============================================================================

/**
 * Get Gemini API key from environment variables
 * 
 * OWASP: Never hardcode secrets, use environment variables
 * 
 * @returns {string|null} API key or null if not configured
 */
export function getGeminiApiKey() {
    const key = import.meta.env.VITE_GEMINI_API_KEY;

    if (!key || key === 'your_api_key_here' || key === 'YOUR_GEMINI_API_KEY_HERE') {
        console.warn(
            '[Security] Gemini API key not configured. ' +
            'Set VITE_GEMINI_API_KEY in your .env file.'
        );
        return null;
    }

    // Basic validation - Gemini keys start with 'AI' and are ~39 chars
    if (key.length < 30) {
        console.warn('[Security] Gemini API key appears to be invalid (too short)');
        return null;
    }

    return key;
}

/**
 * Check if API key is configured
 * @returns {boolean}
 */
export function isApiKeyConfigured() {
    return getGeminiApiKey() !== null;
}

// =============================================================================
// ERROR HANDLING
// =============================================================================

/**
 * Create a user-friendly error response for rate limiting
 * Mimics HTTP 429 Too Many Requests
 * 
 * @param {number} retryAfterMs 
 * @returns {Object}
 */
export function createRateLimitError(retryAfterMs) {
    return {
        status: 429,
        statusText: 'Too Many Requests',
        retryAfter: Math.ceil(retryAfterMs / 1000),
        message: 'Rate limit exceeded. Please slow down and try again shortly.'
    };
}

/**
 * Create a user-friendly error response for validation errors
 * Mimics HTTP 400 Bad Request
 * 
 * @param {string} message 
 * @returns {Object}
 */
export function createValidationError(message) {
    return {
        status: 400,
        statusText: 'Bad Request',
        message: message || 'Invalid input provided.'
    };
}
