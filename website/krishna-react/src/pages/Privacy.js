import React from 'react';
import Navbar from '../components/Navbar';
import './Privacy.css';

function Privacy() {
    return (
        <div className="page-container privacy-page">
            <Navbar />

            <section className="privacy-hero">
                <div className="container">
                    <div className="privacy-hero-content">
                        <span className="section-badge">法的情報</span>
                        <h1 className="privacy-title">
                            プライバシー <span className="gradient-text">ポリシー</span>
                        </h1>
                        <p className="privacy-subtitle">
                            最終更新日: 2026年2月6日
                        </p>
                    </div>
                </div>
            </section>

            <section className="privacy-content">
                <div className="container">
                    <div className="privacy-wrapper">
                        <div className="privacy-section">
                            <h2>はじめに</h2>
                            <p>
                                「クリシュナと話す」へようこそ。私たちはあなたのプライバシーを尊重し、あなたの個人データを保護することに尽力しています。
                                このプライバシーポリシーは、あなたが当社のウェブサイトを訪問した際の個人データの取り扱い方法と、あなたのプライバシーに関する権利について説明するものです。
                            </p>
                        </div>

                        <div className="privacy-section">
                            <h2>収集する情報</h2>
                            <p>当社は、あなたに関する以下の種類の個人データを収集、使用、保存、および転送する場合があります：</p>
                            <ul>
                                <li><strong>本人確認データ：</strong> 名、姓、ユーザー名</li>
                                <li><strong>連絡先データ：</strong> メールアドレス</li>
                                <li><strong>技術的データ：</strong> IPアドレス、ブラウザの種類、デバイス情報</li>
                                <li><strong>利用データ：</strong> 当社のウェブサイトおよびサービスの利用方法に関する情報</li>
                                <li><strong>対話データ：</strong> AIとの質問および対話の内容</li>
                            </ul>
                        </div>

                        <div className="privacy-section">
                            <h2>情報の利用方法</h2>
                            <p>当社は、以下の目的であなたの個人データを使用します：</p>
                            <ul>
                                <li>サービスの提供および維持のため</li>
                                <li>ユーザー体験の向上およびパーソナライズのため</li>
                                <li>アップデートやサポートに関する連絡のため</li>
                                <li>利用パターンの分析およびAIの改善のため</li>
                                <li>プラットフォームのセキュリティ確保のため</li>
                            </ul>
                        </div>

                        <div className="privacy-section">
                            <h2>データのセキュリティ</h2>
                            <p>
                                当社は、あなたの個人データが誤って紛失したり、不正に使用されたり、アクセスされたりすることを防ぐために、適切なセキュリティ対策を講じています。
                                あなたの個人データへのアクセスは、業務上知る必要のある従業員、代理人、請負業者、およびその他の第三者に限定しています。
                            </p>
                        </div>

                        <div className="privacy-section">
                            <h2>あなたのプライバシー権</h2>
                            <p>データ保護法に基づき、次のような権利があります：</p>
                            <ul>
                                <li><strong>アクセス権：</strong> 個人データのコピーを請求する権利</li>
                                <li><strong>訂正権：</strong> 不正確なデータの修正を請求する権利</li>
                                <li><strong>消去権：</strong> 個人データの削除を請求する権利</li>
                                <li><strong>処理の制限権：</strong> 処理の制限を請求する権利</li>
                                <li><strong>データポータビリティ権：</strong> データの転送を請求する権利</li>
                                <li><strong>異議申立権：</strong> 個人データの処理に対して異議を申し立てる権利</li>
                            </ul>
                        </div>

                        <div className="privacy-section">
                            <h2>クッキー（Cookie）</h2>
                            <p>
                                当社は、サービス上の活動を追跡し、特定の情報を保持するために、クッキーおよび同様の追跡技術を使用しています。
                                ブラウザの設定により、すべてのクッキーを拒否したり、クッキーが送信される際に表示したりするように指示できます。
                            </p>
                        </div>

                        <div className="privacy-section">
                            <h2>第三者サービス</h2>
                            <p>当社は、以下を含むサービスの提供を促進するために、第三者の企業および個人を雇用する場合があります：</p>
                            <ul>
                                <li>クラウドホスティングプロバイダー</li>
                                <li>アナリティクスサービス</li>
                                <li>AIおよび機械学習プラットフォーム</li>
                                <li>メールサービスプロバイダー</li>
                            </ul>
                            <p>これらの第三者は、当社に代わって特定のタスクを実行するためにのみあなたの個人データにアクセスし、他の目的で開示または使用しない義務を負っています。</p>
                        </div>

                        <div className="privacy-section">
                            <h2>お子様のプライバシー</h2>
                            <p>
                                当社のサービスは、13歳未満のお子様を対象としていません。当社は、13歳未満のお子様から意図的に個人を特定できる情報を収集することはありません。
                                保護者の方でお子様が個人データを提供したことを知った場合は、当社までご連絡ください。
                            </p>
                        </div>

                        <div className="privacy-section">
                            <h2>本プライバシーポリシーの変更</h2>
                            <p>
                                当社は、プライバシーポリシーを随時更新する場合があります。変更があった場合は、
                                このページに新しいプライバシーポリシーを掲載し、上部の「最終更新日」を更新することで通知いたします。
                            </p>
                        </div>

                        <div className="privacy-section">
                            <h2>お問い合わせ</h2>
                            <p>本プライバシーポリシーについてご質問がある場合は、以下までご連絡ください：</p>
                            <ul>
                                <li>メール: privacy@talktokrishna.com</li>
                                <li>ウェブサイト: <a href="/contact">お問い合わせフォーム</a></li>
                            </ul>
                        </div>

                        <div className="privacy-footer-note">
                            <p>
                                <strong>注记:</strong> あなたの精神的な旅は、あなたにとって個人的なものです。私たちはあなたのプライバシーを尊重し、
                                マーケティング目的であなたの会話や個人情報を第三者と共有することはありません。
                            </p>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    );
}

export default Privacy;
