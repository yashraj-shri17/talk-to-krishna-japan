import React, { useState } from 'react';
import Navbar from '../components/Navbar';
import './Contact.css';

function Contact() {
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        subject: '',
        message: ''
    });

    const handleSubmit = (e) => {
        e.preventDefault();
        alert('Thank you for contacting us! We will get back to you soon.');
        setFormData({ name: '', email: '', subject: '', message: '' });
    };

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    return (
        <div className="page-container contact-page">
            <Navbar />

            <section className="contact-hero">
                <div className="container">
                    <div className="contact-hero-content">
                        <span className="section-badge">お問い合わせ</span>
                        <h1 className="contact-title">
                            私たちは <span className="gradient-text">サポート</span> します
                        </h1>
                        <p className="contact-subtitle">
                            ご質問やフィードバックがありますか？ぜひお聞かせください。
                        </p>
                    </div>
                </div>
            </section>

            <section className="contact-content">
                <div className="container">
                    <div className="contact-grid">
                        <div className="contact-info">
                            <h2>連絡先情報</h2>
                            <p className="info-subtitle">フォームを記入してください。24時間以内にチームからご連絡いたします。</p>

                            <div className="contact-cards">
                                <div className="contact-card">
                                    <div className="contact-icon">📧</div>
                                    <h3>Email</h3>
                                    <p>support@talktokrishna.com</p>
                                </div>

                                <div className="contact-card">
                                    <div className="contact-icon">💬</div>
                                    <h3>Live Chat</h3>
                                    <p>Available 24/7</p>
                                </div>

                                <div className="contact-card">
                                    <div className="contact-icon">🌐</div>
                                    <h3>ソーシャルメディア</h3>
                                    <div className="social-links">
                                        <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" className="social-link">Twitter</a>
                                        <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" className="social-link">LinkedIn</a>
                                        <a href="https://instagram.com" target="_blank" rel="noopener noreferrer" className="social-link">Instagram</a>
                                    </div>
                                </div>
                            </div>

                            <div className="faq-section">
                                <h3>よくある質問</h3>
                                <div className="faq-item">
                                    <strong>AIはどのように機能しますか？</strong>
                                    <p>当社のAIはRAGテクノロジーを使用して、700以上の本物のバガヴァッド・ギーターの聖句を検索します。</p>
                                </div>
                                <div className="faq-item">
                                    <strong>利用は無料ですか？</strong>
                                    <p>はい！「クリシュナと話す」はすべてのユーザーが完全に無料で利用できます。</p>
                                </div>
                                <div className="faq-item">
                                    <strong>サポートされている言語は何ですか？</strong>
                                    <p>現在、英語とヒンディー語（ヒングリッシュ）をサポートしています。</p>
                                </div>
                            </div>
                        </div>

                        <div className="contact-form-wrapper">
                            <form onSubmit={handleSubmit} className="contact-form">
                                <h2>メッセージを送る</h2>

                                <div className="form-group">
                                    <label>お名前</label>
                                    <input
                                        type="text"
                                        name="name"
                                        value={formData.name}
                                        onChange={handleChange}
                                        placeholder="アルジュナ"
                                        required
                                    />
                                </div>

                                <div className="form-group">
                                    <label>メールアドレス</label>
                                    <input
                                        type="email"
                                        name="email"
                                        value={formData.email}
                                        onChange={handleChange}
                                        placeholder="you@example.com"
                                        required
                                    />
                                </div>

                                <div className="form-group">
                                    <label>件名</label>
                                    <input
                                        type="text"
                                        name="subject"
                                        value={formData.subject}
                                        onChange={handleChange}
                                        placeholder="どのようなご用件でしょうか？"
                                        required
                                    />
                                </div>

                                <div className="form-group">
                                    <label>メッセージ</label>
                                    <textarea
                                        name="message"
                                        value={formData.message}
                                        onChange={handleChange}
                                        placeholder="ご質問やフィードバックの詳細を入力してください..."
                                        rows="5"
                                        required
                                    ></textarea>
                                </div>

                                <button type="submit" className="btn-premium-primary btn-large">
                                    <span className="btn-icon">📨</span>
                                    メッセージ送信
                                    <span className="btn-arrow">→</span>
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    );
}

export default Contact;
