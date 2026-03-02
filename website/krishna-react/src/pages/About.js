import React from 'react';
import Navbar from '../components/Navbar';
import './About.css';

function About() {
    return (
        <div className="page-container about-page">
            <Navbar />

            <section className="about-hero">
                <div className="container">
                    <div className="about-hero-content">
                        <span className="section-badge">私たちの使命</span>
                        <h1 className="about-title">
                            <span className="gradient-text">古来の知恵</span> と
                            <br />現代のテクノロジーの架け橋
                        </h1>
                        <p className="about-subtitle">
                            AIの力を通じて、バガヴァッド・ギーターの時代を超えた教えをあらゆる人に届けます
                        </p>
                    </div>
                </div>
            </section>

            <section className="vision-section">
                <div className="container">
                    <div className="vision-grid">
                        <div className="vision-content">
                            <h2 className="section-heading">ビジョン</h2>
                            <p className="vision-text">
                                ノイズと混乱に満ちた世界で、明確で倫理的、そして精神的な導きを見つけることは困難です。
                                「クリシュナと話す」は、バガヴァッド・ギーターの時代を超えた知恵を、いつでも、どこでも、リアルタイムで
                                すべての人に届けるというアイデアから生まれました。
                            </p>
                            <p className="vision-text">
                                高度な大規模言語モデル（LLM）と本物の聖典データを組み合わせることで、
                                単に質問に答えるだけでなく、神聖な慈愛と権威を持ってあなたを導く存在を創り上げました。
                            </p>

                            <div className="vision-stats">
                                <div className="vision-stat-card">
                                    <div className="stat-icon">📜</div>
                                    <div className="stat-info">
                                        <div className="stat-value">700+</div>
                                        <div className="stat-desc">本物の聖句</div>
                                    </div>
                                </div>
                                <div className="vision-stat-card">
                                    <div className="stat-icon">🌍</div>
                                    <div className="stat-info">
                                        <div className="stat-value">グローバル</div>
                                        <div className="stat-desc">アクセシビリティ</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="vision-visual">
                            <div className="visual-card">
                                <div className="visual-icon">🧠</div>
                                <div className="visual-plus">+</div>
                                <div className="visual-icon">📖</div>
                                <div className="visual-equals">=</div>
                                <div className="visual-icon">✨</div>
                            </div>
                            <p className="visual-caption">AIと古代聖典の融合</p>
                        </div>
                    </div>
                </div>
            </section>

            <section className="tech-section">
                <div className="container">
                    <div className="section-header-center">
                        <span className="section-badge">テクノロジー</span>
                        <h2 className="section-heading-center">
                            <span className="gradient-text">最先端AI</span> による駆動
                        </h2>
                        <p className="section-desc">
                            当社のプラットフォームは、複数の高度なテクノロジーを組み合わせて、本物の精神的な導きを提供します
                        </p>
                    </div>

                    <div className="tech-grid">
                        <div className="tech-card">
                            <div className="tech-icon-wrapper">
                                <div className="tech-icon">🔍</div>
                            </div>
                            <h3>RAGアーキテクチャ</h3>
                            <p>検索拡張生成により、すべての回答がバガヴァッド・ギーターの実際の聖句に基づいていることを保証します。</p>
                            <ul className="tech-features">
                                <li>セマンティック検索</li>
                                <li>文脈を考慮した検索</li>
                                <li>検証済みの情報源</li>
                            </ul>
                        </div>

                        <div className="tech-card featured-tech">
                            <div className="tech-badge">コアテクノロジー</div>
                            <div className="tech-icon-wrapper">
                                <div className="tech-icon">🎙️</div>
                            </div>
                            <h3>ニューラル音声</h3>
                            <p>最先端のテキスト読み上げモデルにより、実物に近い穏やかな音声体験を提供します。</p>
                            <ul className="tech-features">
                                <li>自然な抑揚</li>
                                <li>感情表現</li>
                                <li>リアルタイム生成</li>
                            </ul>
                        </div>

                        <div className="tech-card">
                            <div className="tech-icon-wrapper">
                                <div className="tech-icon">🧠</div>
                            </div>
                            <h3>LLMプロセッシング</h3>
                            <p>高度な言語モデルが、単なるキーワードではなく、あなたの質問の背後にある意図を理解します。</p>
                            <ul className="tech-features">
                                <li>文脈の理解</li>
                                <li>多言語サポート</li>
                                <li>パーソナライズされた回答</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </section>

            <section className="values-section">
                <div className="container">
                    <div className="section-header-center">
                        <span className="section-badge">私たちの価値観</span>
                        <h2 className="section-heading-center">
                            <span className="gradient-text">信頼性</span> に基づく構築
                        </h2>
                    </div>

                    <div className="values-grid">
                        <div className="value-card">
                            <div className="value-number">01</div>
                            <h3>聖典の正確性</h3>
                            <p>すべての回答は本物のバガヴァッド・ギーターの聖句に照らして検証されます。教えを偽造したり誤解させたりすることはありません。</p>
                        </div>
                        <div className="value-card">
                            <div className="value-number">02</div>
                            <h3>アクセシビリティ</h3>
                            <p>背景や専門知識に関係なく、すべての人に古来の知恵を届けます。</p>
                        </div>
                        <div className="value-card">
                            <div className="value-number">03</div>
                            <h3>プライバシー</h3>
                            <p>あなたの精神的な旅は個人的なものです。私たちはあなたのプライバシーを尊重し、対話内容を共有することはありません。</p>
                        </div>
                        <div className="value-card">
                            <div className="value-number">04</div>
                            <h3>イノベーション</h3>
                            <p>より良く、より意味のある精神的な導きを提供するために、継続的にテクノロジーを改善しています。</p>
                        </div>
                    </div>
                </div>
            </section>

            <section className="founder-section">
                <div className="container">
                    <div className="section-header-center">
                        <span className="section-badge">リーダーシップ</span>
                        <h2 className="section-heading-center">
                            <span className="gradient-text">創業者</span> の紹介
                        </h2>
                    </div>

                    <div className="founder-card">
                        <div className="founder-image-wrapper">
                            <div className="founder-image-bg"></div>
                            <img
                                src="/founder.jpg"
                                alt="Abhishek Chola - Founder & CEO"
                                className="founder-image"
                                onError={(e) => {
                                    e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="400" height="400"%3E%3Crect fill="%23f0f0f0" width="400" height="400"/%3E%3Ctext fill="%23999" font-family="sans-serif" font-size="24" dy="10.5" font-weight="bold" x="50%25" y="50%25" text-anchor="middle"%3EFounder%3C/text%3E%3C/svg%3E';
                                }}
                            />
                        </div>
                        <div className="founder-content">
                            <h3 className="founder-name">Abhishek Chola</h3>
                            <p className="founder-title">創業者 兼 CEO, Just Learn</p>
                            <div className="founder-divider"></div>
                            <p className="founder-bio">
                                教育と未来のスキルの民主化を使命とする、グローバルなEdTechおよびSkillTechのイノベーター。
                                Abhishekは複数の国にわたる戦略的イニシアチブをリードし、教育とテクノロジーのコラボレーションを促進しています。
                            </p>
                            <p className="founder-bio">
                                彼のリーダーシップは、AI、AR/VR、イマーシブ・テクノロジーを活用したスケーラブルな学習ソリューションに焦点を当て、
                                世界中の学習者に最先端のイノベーションをもたらしています。
                            </p>
                            <div className="founder-highlights">
                                <div className="highlight-item">
                                    <div className="highlight-icon">🌍</div>
                                    <div className="highlight-text">
                                        <strong>グローバルな影響</strong>
                                        <span>多国籍展開</span>
                                    </div>
                                </div>
                                <div className="highlight-item">
                                    <div className="highlight-icon">🚀</div>
                                    <div className="highlight-text">
                                        <strong>イノベーションリーダー</strong>
                                        <span>AI, AR/VR, 没入型技術</span>
                                    </div>
                                </div>
                                <div className="highlight-item">
                                    <div className="highlight-icon">🎓</div>
                                    <div className="highlight-text">
                                        <strong>EdTechパイオニア</strong>
                                        <span>教育の民主化</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            <section className="cta-about">
                <div className="container">
                    <div className="cta-about-card">
                        <h2>神聖な知恵を体験する準備はできましたか？</h2>
                        <p>バガヴァッド・ギーターからのAI導きで、あなたの精神的な旅を始めましょう</p>
                        <button className="btn-premium-primary btn-large" onClick={() => window.location.href = '/chat'}>
                            <span className="btn-icon">🕉️</span>
                            今すぐクリシュナと話す
                            <span className="btn-arrow">→</span>
                        </button>
                    </div>
                </div>
            </section>
        </div>
    );
}

export default About;
