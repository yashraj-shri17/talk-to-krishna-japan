import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import ThemeToggle from './ThemeToggle';
import './Navbar.css';

function Navbar() {
    const location = useLocation();
    const { user } = useAuth();
    const [scrolled, setScrolled] = useState(false);
    const [isOpen, setIsOpen] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 50);
        };

        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const toggleMenu = () => setIsOpen(!isOpen);

    const closeMenu = () => setIsOpen(false);

    const getInitials = (name) => {
        return name
            .split(' ')
            .map(word => word[0])
            .join('')
            .toUpperCase()
            .slice(0, 2);
    };

    return (
        <nav className={`navbar glass ${scrolled ? 'scrolled' : ''}`}>
            <div className="container navbar-content">
                <Link to="/" className="navbar-logo" onClick={closeMenu}>
                    <span className="logo-icon">🕉️</span>
                    <span className="logo-text">クリシュナと話す</span>
                </Link>

                <div className="mobile-toggle" onClick={toggleMenu}>
                    <div className={`hamburger ${isOpen ? 'active' : ''}`}>
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>

                <div className={`navbar-links ${isOpen ? 'active' : ''}`}>
                    <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`} onClick={closeMenu}>ホーム</Link>
                    <Link to="/about" className={`nav-link ${location.pathname === '/about' ? 'active' : ''}`} onClick={closeMenu}>私たちについて</Link>
                    <Link to="/contact" className={`nav-link ${location.pathname === '/contact' ? 'active' : ''}`} onClick={closeMenu}>お問い合わせ</Link>
                    <Link to="/privacy" className={`nav-link ${location.pathname === '/privacy' ? 'active' : ''}`} onClick={closeMenu}>プライバシー</Link>

                    {user ? (
                        <>
                            {(user.has_chat_access || user.role === 'admin') && (
                                <Link to="/chat" className={`nav-link ${location.pathname === '/chat' ? 'active' : ''}`} onClick={closeMenu}>チャット</Link>
                            )}
                            {user.role === 'admin' && (
                                <Link to="/admin" className={`nav-link ${location.pathname === '/admin' ? 'active' : ''} admin-link-pill`} onClick={closeMenu}>管理パネル</Link>
                            )}
                            <Link to="/profile" className="nav-link profile-link" onClick={closeMenu}>
                                <div className="nav-avatar">
                                    {getInitials(user.name)}
                                </div>
                                <span>{user.name.split(' ')[0]}</span>
                            </Link>
                        </>
                    ) : (
                        <>
                            <Link to="/login" className={`nav-link ${location.pathname === '/login' ? 'active' : ''}`} onClick={closeMenu}>ログイン</Link>
                            <Link to="/signup" className="btn-primary" onClick={closeMenu}>新規登録</Link>
                        </>
                    )}

                    <ThemeToggle />
                </div>
            </div>
        </nav>
    );
}

export default Navbar;
