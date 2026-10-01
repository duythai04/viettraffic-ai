CREATE DATABASE IF NOT EXISTS viettraffic_ai
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE viettraffic_ai;



 -- 1. user

CREATE TABLE users (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,

    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,

    role ENUM('USER', 'ADMIN') NOT NULL DEFAULT 'USER',

    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
);



-- 2. conversations

CREATE TABLE conversations (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,

    user_id BIGINT UNSIGNED NOT NULL,

    title VARCHAR(255),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT fk_conversation_user
        FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE
);

CREATE INDEX idx_conversations_user
ON conversations(user_id);



-- 3. msssages

CREATE TABLE messages (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,

    conversation_id BIGINT UNSIGNED NOT NULL,

    role ENUM('USER', 'ASSISTANT') NOT NULL,

    content TEXT NOT NULL,

    -- Phân tích câu hỏi của Legal RAG
    vehicle VARCHAR(50),
    topic VARCHAR(100),
    intent VARCHAR(50),

    -- Kết quả kiểm tra RAG
    validation_status VARCHAR(30),
    validation_confidence VARCHAR(30),

    needs_clarification BOOLEAN DEFAULT FALSE,
    clarification_question TEXT,

    model_name VARCHAR(100),

    prompt_tokens INT UNSIGNED,
    completion_tokens INT UNSIGNED,
    total_tokens INT UNSIGNED,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_message_conversation
        FOREIGN KEY (conversation_id)
        REFERENCES conversations(id)
        ON DELETE CASCADE
);

CREATE INDEX idx_messages_conversation
ON messages(conversation_id);

CREATE INDEX idx_messages_created
ON messages(created_at);




-- 4. legal documents

CREATE TABLE legal_documents (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,

    document_type VARCHAR(50) NOT NULL,

    document_number VARCHAR(100),

    title VARCHAR(500) NOT NULL,

    filename VARCHAR(255),

    source_path VARCHAR(500),

    issued_date DATE,
    effective_date DATE,
    expiry_date DATE,

    status ENUM(
        'ACTIVE',
        'PARTIALLY_ACTIVE',
        'EXPIRED',
        'REPEALED',
        'UNKNOWN'
    ) DEFAULT 'UNKNOWN',

    issuing_authority VARCHAR(255),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
);

CREATE INDEX idx_legal_document_number
ON legal_documents(document_number);

CREATE INDEX idx_legal_document_status
ON legal_documents(status);


-- 5. message sources
CREATE TABLE message_sources (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,

    message_id BIGINT UNSIGNED NOT NULL,
    legal_document_id BIGINT UNSIGNED,

    source_label VARCHAR(20),

    page_number INT,

    article_number VARCHAR(20),
    clause_number VARCHAR(20),
    point_label VARCHAR(20),

    retrieval_type VARCHAR(50),

    content TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_source_message
        FOREIGN KEY (message_id)
        REFERENCES messages(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_source_document
        FOREIGN KEY (legal_document_id)
        REFERENCES legal_documents(id)
        ON DELETE SET NULL
);

CREATE INDEX idx_message_sources_message
ON message_sources(message_id);


-- 6. feedbacks

CREATE TABLE feedbacks (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,

    user_id BIGINT UNSIGNED NOT NULL,
    message_id BIGINT UNSIGNED NOT NULL,

    rating ENUM('LIKE', 'DISLIKE') NOT NULL,

    comment TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_feedback_user
        FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_feedback_message
        FOREIGN KEY (message_id)
        REFERENCES messages(id)
        ON DELETE CASCADE,

    UNIQUE KEY uk_feedback_user_message (
        user_id,
        message_id
    )
);


-- 7. rag log

CREATE TABLE rag_logs (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,

    message_id BIGINT UNSIGNED,

    query TEXT NOT NULL,

    vehicle VARCHAR(50),
    topic VARCHAR(100),
    intent VARCHAR(50),

    retrieved_count INT DEFAULT 0,

    validation_status VARCHAR(30),
    validation_confidence VARCHAR(30),

    penalty_required BOOLEAN DEFAULT FALSE,
    penalty_complete BOOLEAN DEFAULT FALSE,

    needs_clarification BOOLEAN DEFAULT FALSE,

    model_name VARCHAR(100),

    prompt_tokens INT UNSIGNED DEFAULT 0,
    completion_tokens INT UNSIGNED DEFAULT 0,
    total_tokens INT UNSIGNED DEFAULT 0,

    processing_time_ms INT UNSIGNED,

    error_message TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_rag_log_message
        FOREIGN KEY (message_id)
        REFERENCES messages(id)
        ON DELETE SET NULL
);

CREATE INDEX idx_rag_logs_created
ON rag_logs(created_at);

CREATE INDEX idx_rag_logs_topic
ON rag_logs(topic);