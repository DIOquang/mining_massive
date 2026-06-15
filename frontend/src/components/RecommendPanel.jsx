export default function RecommendPanel({ recommendations, loading, error, clickedItems }) {
  if (loading) {
    return (
      <div className="recommend-state">
        <div className="ai-thinking">
          <div className="ai-orb" />
          <p className="ai-text">AI đang phân tích sở thích của bạn...</p>
          <div className="loading-dots">
            <span /><span /><span />
          </div>
        </div>
        <div className="recommend-skeletons">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="rec-skeleton" style={{ animationDelay: `${i * 0.1}s` }} />
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="recommend-state error-state">
        <div className="error-icon">⚠️</div>
        <p className="error-text">{error}</p>
      </div>
    )
  }

  if (recommendations.length === 0) {
    return (
      <div className="recommend-empty">
        <div className="empty-visual">
          <div className="empty-circles">
            <div className="empty-circle c1" />
            <div className="empty-circle c2" />
            <div className="empty-circle c3" />
          </div>
          <span className="empty-icon">🤖</span>
        </div>
        <h3 className="empty-title">Hệ thống AI sẵn sàng</h3>
        <p className="empty-subtitle">
          Hãy bấm vào sản phẩm bạn quan tâm bên trái để hệ thống Two-Tower
          bắt đầu cá nhân hóa gợi ý cho bạn.
        </p>
        {clickedItems.length === 0 && (
          <div className="empty-hint">
            <span className="hint-arrow">←</span>
            <span>Chọn sản phẩm đầu tiên</span>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="recommend-list">
      {/* Session context */}
      {clickedItems.length > 0 && (
        <div className="session-context">
          <span className="context-label">Dựa trên:</span>
          <div className="context-items">
            {clickedItems.map(item => (
              <span key={item.item_id} className="context-chip">
                {item.title?.split(' ').slice(0, 3).join(' ') || item.item_id}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Danh sách gợi ý */}
      {recommendations.map((rec, idx) => (
        <div
          key={rec.item_id}
          className="rec-card"
          style={{ animationDelay: `${idx * 0.08}s` }}
        >
          <div className="rec-rank">#{idx + 1}</div>
          <div className="rec-content">
            <div className="rec-title">{rec.title}</div>
            <div className="rec-explanation">
              <span className="ai-badge">🤖 AI</span>
              {rec.explanation}
            </div>
          </div>
          <div className="rec-score-bar">
            <div
              className="rec-score-fill"
              style={{ width: `${(rec.score * 100).toFixed(0)}%` }}
            />
            <span className="rec-score-label">{(rec.score * 100).toFixed(0)}%</span>
          </div>
        </div>
      ))}
    </div>
  )
}
