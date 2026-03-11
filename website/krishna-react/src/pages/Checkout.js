import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import axios from 'axios';
import { Check, Ticket, ShieldCheck, ArrowRight, X } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { API_ENDPOINTS } from '../config/api';
import './Checkout.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function Checkout() {
    const location = useLocation();
    const navigate = useNavigate();
    const { user, login } = useAuth();

    // Fallback if accessed directly without state
    const selectedPlan = location.state?.plan || {
        name: 'プラン未選択',
        price: '¥0',
        period: '',
        description: ''
    };

    const [couponCode, setCouponCode] = useState('');
    const [appliedCoupon, setAppliedCoupon] = useState(null);
    const [couponError, setCouponError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [checkoutComplete, setCheckoutComplete] = useState(false);

    const basePriceNum = parseInt(selectedPlan.price.replace(/[¥,]/g, '')) || 0;
    const [finalPrice, setFinalPrice] = useState(basePriceNum);

    useEffect(() => {
        if (!location.state?.plan) {
            navigate('/pricing');
        }
    }, [location.state, navigate]);

    const handleApplyCoupon = async () => {
        if (!couponCode.trim()) return;

        setIsLoading(true);
        setCouponError('');

        try {
            const response = await axios.post(`${API_BASE_URL}/api/validate-coupon`, {
                code: couponCode
            });

            if (response.data.success) {
                const coupon = response.data.coupon;
                setAppliedCoupon(coupon);
                setCouponCode('');

                // Calculate discount (Ensuring integer values for Yen)
                let discountAmount = 0;
                if (coupon.discount_type === 'percentage') {
                    discountAmount = Math.round((basePriceNum * coupon.discount_value) / 100);
                } else if (coupon.discount_type === 'fixed_value') {
                    discountAmount = Math.round(coupon.discount_value);
                } else if (coupon.discount_type === 'free_access') {
                    discountAmount = basePriceNum;
                }

                setFinalPrice(Math.max(0, basePriceNum - discountAmount));
            }
        } catch (err) {
            setCouponError(err.response?.data?.error || '無効なクーポンコードです');
            setAppliedCoupon(null);
            setFinalPrice(basePriceNum);
        } finally {
            setIsLoading(false);
        }
    };

    const removeCoupon = () => {
        setAppliedCoupon(null);
        setFinalPrice(basePriceNum);
    };

    const handleCompletePurchase = async () => {
        setIsLoading(true);

        try {
            // Grant chat access in the database
            const response = await axios.post(API_ENDPOINTS.GRANT_ACCESS, {
                user_id: user?.id
            });

            if (response.data.success) {
                // Update local user session so chat works immediately
                if (user) {
                    const updatedUser = { ...user, has_chat_access: true };
                    login(updatedUser); // updates localStorage + context
                }
                setCheckoutComplete(true);
                setTimeout(() => {
                    navigate('/chat');
                }, 3000);
            } else {
                alert('Access grant failed. Please contact support.');
            }
        } catch (err) {
            console.error('Payment/Access error:', err);
            alert('Something went wrong. Please try again or contact support.');
        } finally {
            setIsLoading(false);
        }
    };

    if (checkoutComplete) {
        return (
            <div className="checkout-page">
                <Navbar />
                <div className="checkout-success-container container">
                    <div className="success-content">
                        <div className="success-icon-wrapper">
                            <Check size={48} />
                        </div>
                        <h1>注文が完了しました！</h1>
                        <p>プランの有効化に成功しました。まもなくチャットページに移動します...</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="checkout-page">
            <Navbar />

            <div className="checkout-container container">
                <div className="checkout-layout">
                    {/* Left: Plan Summary */}
                    <div className="checkout-summary">
                        <div className="checkout-card">
                            <h2 className="section-title">ご注文内容</h2>
                            <div className="plan-item">
                                <div className="plan-info">
                                    <h3>{selectedPlan.name}プラン</h3>
                                    <p>{selectedPlan.description}</p>
                                </div>
                                <div className="plan-price-side">
                                    {selectedPlan.price}
                                </div>
                            </div>

                            <hr className="divider" />

                            <div className="price-details">
                                <div className="price-row">
                                    <span>小計</span>
                                    <span>¥{basePriceNum.toLocaleString()}</span>
                                </div>

                                {appliedCoupon && (
                                    <div className="price-row discount">
                                        <span>クーポン適用 ({appliedCoupon.code})</span>
                                        <span>- ¥{(basePriceNum - finalPrice).toLocaleString()}</span>
                                    </div>
                                )}

                                <div className="price-row total">
                                    <span>合計</span>
                                    <span className="final-total">¥{finalPrice.toLocaleString()}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Right: Payment & Options */}
                    <div className="checkout-actions">
                        {/* Coupon Form */}
                        <div className="checkout-card coupon-section">
                            <h3>クーポンを利用する</h3>
                            {!appliedCoupon ? (
                                <div className="coupon-input-group">
                                    <input
                                        type="text"
                                        placeholder="クーポンコードを入力"
                                        value={couponCode}
                                        onChange={(e) => setCouponCode(e.target.value.toUpperCase())}
                                        className="coupon-input"
                                    />
                                    <button
                                        onClick={handleApplyCoupon}
                                        disabled={isLoading || !couponCode}
                                        className="apply-btn"
                                    >
                                        適用
                                    </button>
                                </div>
                            ) : (
                                <div className="applied-badge">
                                    <Ticket size={18} />
                                    <span>{appliedCoupon.code} が適用されました</span>
                                    <button onClick={removeCoupon} className="remove-btn">
                                        <X size={14} />
                                    </button>
                                </div>
                            )}
                            {couponError && <p className="coupon-error">{couponError}</p>}
                        </div>

                        {/* Payment Button */}
                        <div className="checkout-card payment-section">
                            <div className="security-tag">
                                <ShieldCheck size={16} />
                                <span>安全なSSL暗号化通信</span>
                            </div>

                            <button
                                className="complete-btn"
                                onClick={handleCompletePurchase}
                                disabled={isLoading}
                            >
                                {isLoading ? '処理中...' : '決済を完了する'}
                                {!isLoading && <ArrowRight size={20} />}
                            </button>

                            <p className="payment-note">
                                ボタンを押すことで、当社の利用規約に同意したものとみなされます。
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Checkout;
