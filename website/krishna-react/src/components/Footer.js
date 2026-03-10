import React from 'react';
import { Link } from 'react-router-dom';
import './Footer.css';

const Footer = () => {
    return (
        <footer className="footer-premium">
            <div className="container">
                <div className="footer-content">
                    <div className="footer-brand">
                        <div className="footer-logo">
                            <span className="logo-icon">🕉️</span>
                            <span>クリシュナと話す</span>
                        </div>
                        <p>現代のテクノロジーと融合した、古来の知恵</p>
                    </div>
                    <div className="footer-links">
                        <Link to="/about">私たちについて</Link>
                        <Link to="/pricing">料金プラン</Link>
                        <Link to="/contact">お問い合わせ</Link>
                        <Link to="/privacy">プライバシー</Link>
                        <Link to="/login">ログイン</Link>
                        <Link to="/signup">新規登録</Link>
                    </div>
                </div>
                <div className="footer-bottom">
                    <p>© 2026 クリシュナと話す. All rights reserved.</p>
                </div>
            </div>
        </footer>
    );
};

export default Footer;
