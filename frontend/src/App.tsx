import { useState } from 'react';
import config from './portfolio.config.json';
import Profile from './features/profile/Profile';
import Rag from './features/rag/Rag';
import './global.css';

type ModuleId = 'profile' | string;

export default function App() {
  const [active, setActive] = useState<ModuleId>('profile');

  const activeModule = config.modules.find(m => m.id === active) as any;

  function renderView() {
    if (active === 'profile') {
      return (
        <Profile
          profile={config.profile}
          modules={config.modules as any}
          onNavigate={(id) => setActive(id)}
        />
      );
    }
    if (active === 'rag') {
      return (
        <Rag
          description={activeModule?.description}
          topic={activeModule?.topic}
          sampleQuestions={activeModule?.['sample questions']}
          stack={activeModule?.stack}
          notices={activeModule?.notices}
          wakeup={activeModule?.wakeup}
        />
      );
    }
    return null;
  }

  return (
    <div className="portfolio-shell">
      <nav className="portfolio-nav">
        <span className="portfolio-nav__wordmark">{config.profile.name}</span>
        <span className="portfolio-nav__divider" />

        <button
          className={`portfolio-nav__tab ${active === 'profile' ? 'portfolio-nav__tab--active' : ''}`}
          onClick={() => setActive('profile')}
        >
          <span className="portfolio-nav__pip" />
          Profile
        </button>

        {config.modules.map(m => {
          const isLive   = m.status === 'live';
          const isActive = active === m.id;
          return (
            <button
              key={m.id}
              className={[
                'portfolio-nav__tab',
                isActive ? 'portfolio-nav__tab--active'  : '',
                !isLive  ? 'portfolio-nav__tab--disabled' : '',
              ].join(' ')}
              onClick={() => isLive && setActive(m.id)}
              disabled={!isLive}
            >
              <span className="portfolio-nav__pip" />
              {m.label}
              {!isLive && <span className="portfolio-nav__badge">soon</span>}
            </button>
          );
        })}
      </nav>

      <div className="portfolio-view">
        {renderView()}
      </div>
    </div>
  );
}