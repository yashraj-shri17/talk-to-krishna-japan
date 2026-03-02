import React, { useEffect, useRef } from 'react';
import Navbar from '../components/Navbar';
import { useNavigate } from 'react-router-dom';
import './Home.css';

function Home() {
    const navigate = useNavigate();
    const observerRef = useRef(null);

    useEffect(() => {
        // Intersection Observer for scroll animations
        observerRef.current = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('animate-in');
                    }
                });
            },
            { threshold: 0.1 }
        );

        // Observe all animated elements
        const animatedElements = document.querySelectorAll('.fade-in-section');
        animatedElements.forEach((el) => observerRef.current.observe(el));

        return () => {
            if (observerRef.current) {
                observerRef.current.disconnect();
            }
        };
    }, []);

    return (
        <div className="home-page">
            <Navbar />

            {/* Floating Elements Background */}
            <div className="floating-elements">
                <div className="float-element om-symbol">🕉️</div>
                <div className="float-element lotus">🪷</div>
                <div className="float-element peacock">🦚</div>
                <div className="float-element om-symbol-2">ॐ</div>
            </div>

            {/* Hero Section */}
            <section className="hero-section-premium">
                <div className="container hero-grid">
                    <div className="hero-content-left">
                        <div className="badge-premium">
                            <span className="badge-dot"></span>
                            AIを活用した精神的な導き
                        </div>

                        <h1 className="hero-headline-premium">
                            古来の知恵、
                            <br />
                            <span className="gradient-text-animated">現代の声</span>
                        </h1>

                        <p className="hero-description-premium">
                            最先端のAI音声技術を通じて、バガヴァッド・ギーターの時代を超えた教えを体験してください。
                            クリシュナとのリアルタイムな対話で、パーソナライズされた精神的な導きを得られます。
                        </p>

                        <div className="hero-cta-group">
                            <button className="btn-premium-primary" onClick={() => navigate('/chat')}>
                                <span className="btn-icon">🎙️</span>
                                対話を始める
                                <span className="btn-arrow">→</span>
                            </button>
                            <button className="btn-premium-secondary" onClick={() => navigate('/about')}>
                                <span className="btn-icon">📖</span>
                                詳しく知る
                            </button>
                        </div>

                        <div className="stats-row">
                            <div className="stat-item">
                                <div className="stat-number">700+</div>
                                <div className="stat-label">聖句(シュローカ)</div>
                            </div>
                            <div className="stat-divider"></div>
                            <div className="stat-item">
                                <div className="stat-number">24/7</div>
                                <div className="stat-label">利用可能</div>
                            </div>
                            <div className="stat-divider"></div>
                            <div className="stat-item">
                                <div className="stat-number">∞</div>
                                <div className="stat-label">知恵</div>
                            </div>
                        </div>
                    </div>

                    <div className="hero-visual-right">
                        {/* 3D Orb Illustration */}
                        <div className="orb-container-3d">
                            <div className="orb-main">
                                <div className="orb-inner-glow"></div>
                                <div className="orb-particles">
                                    {[...Array(20)].map((_, i) => (
                                        <div key={i} className={`particle p-${i}`}></div>
                                    ))}
                                </div>
                                <div className="orb-rings">
                                    <div className="ring ring-1"></div>
                                    <div className="ring ring-2"></div>
                                    <div className="ring ring-3"></div>
                                </div>
                                <div className="orb-center-icon">🕉️</div>
                            </div>

                            {/* Floating Cards */}
                            <div className="floating-card card-1">
                                <div className="card-icon">🎯</div>
                                <div className="card-content">
                                    <strong>即座の回答</strong>
                                    <span>リアルタイムの導き</span>
                                </div>
                            </div>

                            <div className="floating-card card-2">
                                <div className="card-icon">🧘</div>
                                <div className="card-content">
                                    <strong>精神的成長</strong>
                                    <span>個人的な旅</span>
                                </div>
                            </div>

                            <div className="floating-card card-3">
                                <div className="card-icon">💬</div>
                                <div className="card-content">
                                    <strong>ボイスチャット</strong>
                                    <span>自然な対話</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Features Section - Premium Design */}
            <section className="features-section-premium fade-in-section">
                <div className="container">
                    <div className="section-header-premium">
                        <span className="section-badge">特徴</span>
                        <h2 className="section-title-premium">
                            <span className="gradient-text">クリシュナと話す</span> を選ぶ理由
                        </h2>
                        <p className="section-subtitle">
                            古来の知恵と現代のAIを組み合わせ、あなたにぴったりの精神的な導きを提供します
                        </p>
                    </div>

                    <div className="features-grid-premium">
                        <div className="feature-card-premium">
                            <div className="feature-icon-wrapper">
                                <div className="feature-icon-bg"></div>
                                <div className="feature-icon">🎙️</div>
                            </div>
                            <h3>AI音声対話</h3>
                            <p>日本語や英語で自然に話しかければ、ニューラルTTS技術による本物の音声で回答を受け取れます。</p>
                            <div className="feature-tags">
                                <span className="tag">リアルタイム</span>
                                <span className="tag">自然</span>
                            </div>
                        </div>

                        <div className="feature-card-premium featured">
                            <div className="featured-badge">最も人気</div>
                            <div className="feature-icon-wrapper">
                                <div className="feature-icon-bg"></div>
                                <div className="feature-icon">📜</div>
                            </div>
                            <h3>正確な聖典引用</h3>
                            <p>すべての回答は、先進的なRAGアーキテクチャを使用して、バガヴァッド・ギーターの本物の聖句に基づいています。</p>
                            <div className="feature-tags">
                                <span className="tag">検証済み</span>
                                <span className="tag">本物</span>
                            </div>
                        </div>

                        <div className="feature-card-premium">
                            <div className="feature-icon-wrapper">
                                <div className="feature-icon-bg"></div>
                                <div className="feature-icon">✨</div>
                            </div>
                            <h3>パーソナライズされた知恵</h3>
                            <p>あなたの特定の人生の状況、感情の状態、精神的な旅に合わせたアドバイスを得られます。</p>
                            <div className="feature-tags">
                                <span className="tag">カスタム</span>
                                <span className="tag">文脈に応じた</span>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* How It Works Section */}
            <section className="how-it-works-section fade-in-section">
                <div className="container">
                    <div className="section-header-premium">
                        <span className="section-badge">プロセス</span>
                        <h2 className="section-title-premium">
                            使いかたの <span className="gradient-text">流れ</span>
                        </h2>
                    </div>

                    <div className="steps-container">
                        <div className="step-card">
                            <div className="step-number">01</div>
                            <div className="step-icon">🎤</div>
                            <h3>質問を話す</h3>
                            <p>人生、ダルマ、カルマ、あるいはあなたが求める精神的な導きについて何でも聞いてください</p>
                        </div>

                        <div className="step-connector"></div>

                        <div className="step-card">
                            <div className="step-number">02</div>
                            <div className="step-icon">🧠</div>
                            <h3>AIが処理</h3>
                            <p>当社のRAGシステムが700以上の聖句を検索し、最適な回答を見つけ出します</p>
                        </div>

                        <div className="step-connector"></div>

                        <div className="step-card">
                            <div className="step-number">03</div>
                            <div className="step-icon">🔊</div>
                            <h3>知恵を受け取る</h3>
                            <p>ギーターに基づく本物の導きを、クリシュナの声で聴くことができます</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="cta-section-premium fade-in-section">
                <div className="container">
                    <div className="cta-card-premium">
                        <div className="cta-content">
                            <h2>精神的な旅を始める準備はできましたか？</h2>
                            <p>AIを活用した対話を通じて知恵と導きを求める数千の人々に加わりましょう</p>
                            <button className="btn-premium-primary btn-large" onClick={() => navigate('/chat')}>
                                <span className="btn-icon">🕉️</span>
                                クリシュナと話し始める
                                <span className="btn-arrow">→</span>
                            </button>
                        </div>
                        <div className="cta-decoration">
                            <div className="decoration-circle c1"></div>
                            <div className="decoration-circle c2"></div>
                            <div className="decoration-circle c3"></div>
                        </div>
                    </div>
                </div>
            </section>




        </div>
    );
}

export default Home;
