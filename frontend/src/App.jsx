import { useState, useCallback } from 'react'
import ProductGrid from './components/ProductGrid'
import RecommendPanel from './components/RecommendPanel'
import './App.css'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8080'

export default function App() {
  const [clickedItems, setClickedItems] = useState([])
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(false)
  const [fromCache, setFromCache] = useState(false)
  const [error, setError] = useState(null)

  const handleItemClick = useCallback(async (item) => {
    // Thêm vào danh sách session (tránh trùng)
    const newClicked = clickedItems.find(i => i.item_id === item.item_id)
      ? clickedItems
      : [...clickedItems, item]
    setClickedItems(newClicked)

    // Gọi API gợi ý
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${BACKEND_URL}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clicked_item_ids: newClicked.map(i => i.item_id) }),
      })
      if (!res.ok) throw new Error(`Lỗi API: ${res.status}`)
      const data = await res.json()
      setRecommendations(data.recommendations)
      setFromCache(data.from_cache)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [clickedItems])

  const handleReset = () => {
    setClickedItems([])
    setRecommendations([])
    setError(null)
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">⚡</span>
            <span className="logo-text">ShopAI</span>
            <span className="logo-badge">Two-Tower Recommendation</span>
          </div>
          <div className="header-meta">
            {clickedItems.length > 0 && (
              <div className="session-indicator">
                <span className="dot pulse" />
                Phiên hoạt động — {clickedItems.length} sản phẩm đã chọn
              </div>
            )}
            {clickedItems.length > 0 && (
              <button className="btn-reset" onClick={handleReset}>
                Phiên mới
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main layout */}
      <main className="main-layout">
        <section className="panel panel-left">
          <div className="panel-header">
            <h2>🛍️ Cửa hàng</h2>
            <p className="panel-subtitle">Bấm vào sản phẩm bạn quan tâm</p>
          </div>
          <ProductGrid
            onItemClick={handleItemClick}
            clickedItems={clickedItems}
            backendUrl={BACKEND_URL}
          />
        </section>

        <div className="divider" />

        <section className="panel panel-right">
          <div className="panel-header">
            <h2>🤖 Gợi ý AI</h2>
            <div className="panel-subtitle-row">
              <p className="panel-subtitle">
                {recommendations.length > 0
                  ? `Top ${recommendations.length} sản phẩm phù hợp với bạn`
                  : 'Hãy chọn sản phẩm bạn quan tâm bên trái'}
              </p>
              {fromCache && <span className="cache-badge">⚡ Cache</span>}
            </div>
          </div>
          <RecommendPanel
            recommendations={recommendations}
            loading={loading}
            error={error}
            clickedItems={clickedItems}
          />
        </section>
      </main>
    </div>
  )
}
