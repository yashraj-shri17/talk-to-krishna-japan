import React from 'react';
import Navbar from '../components/Navbar';
import { useNavigate } from 'react-router-dom';
import { Check, Star, Zap, ShieldCheck } from 'lucide-react';
import './Pricing.css';

function Pricing() {
    const navigate = useNavigate();

    const plans = [
        {
            name: '1ヶ月',
            price: '¥480',
            period: '/月',
            description: 'スピリチュアルな対話を体験したい方に最適です',
            features: [
                '全てのAIチャット機能',
                '高音質なニューラル音声',
                '24時間365日のアクセス',
                '保存可能な対話履歴'
            ],
            buttonText: '今すぐ始める',
            isPopular: false,
            color: 'var(--blue-glow)'
        },
        {
            name: '3ヶ月',
            price: '¥1,280',
            period: '/3ヶ月',
            description: '精神的な成長を継続的にサポートするおすすめのプランです',
            features: [
                '1ヶ月プランの全機能',
                '約20%の割引価格',
                '優先的なAIレスポンス',
                'カスタマイズされた導き'
            ],
            buttonText: '最も選ばれています',
            isPopular: true,
            color: 'var(--accent-glow)'
        },
        {
            name: '6ヶ月',
            price: '¥2,280',
            period: '/6ヶ月',
            description: '生涯続く学びと深い自己研鑽に取り組む方のためのプランです',
            features: [
                '3ヶ月プランの全機能',
                '最大の割引料金',
                '早期アクセス機能',
                '専任のサポート'
            ],
            buttonText: '最高の価値を選択',
            isPopular: false,
            color: 'var(--purple-glow)'
        }
    ];

    return (
        <div className="pricing-page">
            <Navbar />

            <div className="pricing-container container">
                <header className="pricing-header">
                    <div className="pricing-badge">
                        <Zap size={14} />
                        <span>PRICING</span>
                    </div>
                    <h1 className="pricing-title">
                        魂の <span className="gradient-text-animated">旅を選択</span>
                    </h1>
                    <p className="pricing-subtitle">
                        あなたのニーズに合わせた最適なプランで、<br />
                        内なる平和と精神的な成長への投資を始めましょう。
                    </p>
                </header>

                <div className="pricing-grid">
                    {plans.map((plan, index) => (
                        <div
                            key={index}
                            className={`pricing-card ${plan.isPopular ? 'popular' : ''}`}
                        >
                            {plan.isPopular && <div className="popular-badge">一番人気</div>}

                            <div className="plan-name">{plan.name}</div>

                            <div className="plan-price">
                                <span className="currency">{plan.price}</span>
                                <span className="period">{plan.period}</span>
                            </div>

                            <p className="plan-description">{plan.description}</p>

                            <div className="plan-features">
                                {plan.features.map((feature, fIndex) => (
                                    <div key={fIndex} className="feature-item">
                                        <div className="check-icon">
                                            <Check size={16} />
                                        </div>
                                        <span>{feature}</span>
                                    </div>
                                ))}
                            </div>

                            <button
                                className={`plan-button ${plan.isPopular ? 'btn-premium-primary' : 'btn-premium-secondary'}`}
                                onClick={() => navigate('/checkout', { state: { plan } })}
                            >
                                {plan.buttonText}
                            </button>
                        </div>
                    ))}
                </div>

                <div className="pricing-trust">
                    <div className="trust-item">
                        <ShieldCheck size={24} />
                        <span>安全な決済</span>
                    </div>
                    <div className="trust-item">
                        <Star size={24} />
                        <span>満足度 100%</span>
                    </div>
                    <div className="trust-item">
                        <Zap size={24} />
                        <span>即時アクセス</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Pricing;
