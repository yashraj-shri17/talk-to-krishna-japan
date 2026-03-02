import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import { useNavigate, Link } from 'react-router-dom';
import { API_ENDPOINTS } from '../config/api';
import './Auth.css';

function Signup() {
    const navigate = useNavigate();

    // Form state
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [showPassword, setShowPassword] = useState(false);
    const [passwordStrength, setPasswordStrength] = useState({
        score: 0,
        label: '',
        color: '',
        checks: {
            length: false,
            uppercase: false,
            lowercase: false,
            number: false,
            special: false
        }
    });

    // Calculate password strength in real-time
    useEffect(() => {
        if (!password) {
            setPasswordStrength({
                score: 0,
                label: '',
                color: '',
                checks: {
                    length: false,
                    uppercase: false,
                    lowercase: false,
                    number: false,
                    special: false
                }
            });
            return;
        }

        const checks = {
            length: password.length >= 8,
            uppercase: /[A-Z]/.test(password),
            lowercase: /[a-z]/.test(password),
            number: /\d/.test(password),
            special: /[!@#$%^&*(),.?":{}|<>]/.test(password)
        };

        const score = Object.values(checks).filter(Boolean).length;

        let label = '';
        let color = '';

        if (score === 0) {
            label = '';
            color = '';
        } else if (score <= 2) {
            label = '弱い';
            color = '#ef4444';
        } else if (score === 3) {
            label = '普通';
            color = '#f59e0b';
        } else if (score === 4) {
            label = '良い';
            color = '#3b82f6';
        } else {
            label = '強い';
            color = '#10b981';
        }

        setPasswordStrength({ score, label, color, checks });
    }, [password]);

    const handleSignup = async (e) => {
        e.preventDefault();
        setError('');

        // Client-side validation
        if (passwordStrength.score < 5) {
            setError('パスワードの要件をすべて満たしてください');
            return;
        }

        setLoading(true);

        try {
            const response = await fetch(API_ENDPOINTS.SIGNUP, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ name, email, password }),
            });

            const data = await response.json();

            if (response.ok) {
                // Auto redirect to login after signup
                navigate('/login', { state: { message: 'アカウントが作成されました！ログインしてください。' } });
            } else {
                setError(data.error || '新規登録に失敗しました');
            }
        } catch (err) {
            setError('接続エラーが発生しました。インターネット接続を確認して、もう一度お試しください。');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="page-container auth-page">
            <Navbar />
            <div className="auth-card glass">
                <div className="auth-header">
                    <h2>アカウント作成</h2>
                    <p>クリシュナと共に旅を始めましょう</p>
                </div>

                {error && <div className="error-message">{error}</div>}

                <form onSubmit={handleSignup} className="auth-form">
                    <div className="form-group">
                        <label>フルネーム</label>
                        <input
                            type="text"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="アルジュナ"
                            required
                            disabled={loading}
                            minLength="2"
                        />
                    </div>

                    <div className="form-group">
                        <label>メールアドレス</label>
                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="you@example.com"
                            required
                            disabled={loading}
                        />
                    </div>

                    <div className="form-group">
                        <label>Password</label>
                        <div className="password-input-wrapper">
                            <input
                                type={showPassword ? "text" : "password"}
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="••••••••"
                                required
                                disabled={loading}
                            />
                            <button
                                type="button"
                                className="password-toggle"
                                onClick={() => setShowPassword(!showPassword)}
                                tabIndex="-1"
                            >
                                {showPassword ? '👁️' : '👁️‍🗨️'}
                            </button>
                        </div>

                        {/* Password Strength Meter */}
                        {password && (
                            <div className="password-strength">
                                <div className="strength-bars">
                                    {[1, 2, 3, 4, 5].map((bar) => (
                                        <div
                                            key={bar}
                                            className={`strength-bar ${bar <= passwordStrength.score ? 'active' : ''}`}
                                            style={{
                                                backgroundColor: bar <= passwordStrength.score ? passwordStrength.color : '#e5e7eb'
                                            }}
                                        ></div>
                                    ))}
                                </div>
                                {passwordStrength.label && (
                                    <span className="strength-label" style={{ color: passwordStrength.color }}>
                                        {passwordStrength.label}
                                    </span>
                                )}
                            </div>
                        )}

                        {/* Password Requirements Checklist */}
                        {password && (
                            <div className="password-requirements">
                                <div className={`requirement ${passwordStrength.checks.length ? 'met' : ''}`}>
                                    {passwordStrength.checks.length ? '✓' : '○'} 少なくとも8文字
                                </div>
                                <div className={`requirement ${passwordStrength.checks.uppercase ? 'met' : ''}`}>
                                    {passwordStrength.checks.uppercase ? '✓' : '○'} 大文字1文字以上
                                </div>
                                <div className={`requirement ${passwordStrength.checks.lowercase ? 'met' : ''}`}>
                                    {passwordStrength.checks.lowercase ? '✓' : '○'} 小文字1文字以上
                                </div>
                                <div className={`requirement ${passwordStrength.checks.number ? 'met' : ''}`}>
                                    {passwordStrength.checks.number ? '✓' : '○'} 数字1文字以上
                                </div>
                                <div className={`requirement ${passwordStrength.checks.special ? 'met' : ''}`}>
                                    {passwordStrength.checks.special ? '✓' : '○'} 特殊文字1文字以上
                                </div>
                            </div>
                        )}
                    </div>

                    <button
                        type="submit"
                        className="btn-primary btn-block"
                        disabled={loading || (password && passwordStrength.score < 5)}
                    >
                        {loading ? (
                            <>
                                <span className="spinner"></span>
                                アカウント作成中...
                            </>
                        ) : '新規登録'}
                    </button>
                </form>

                <div className="auth-footer">
                    <p>すでにアカウントをお持ちですか？ <Link to="/login">ログイン</Link></p>
                </div>
            </div>
        </div>
    );
}

export default Signup;
