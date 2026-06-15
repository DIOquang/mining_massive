import { useEffect, useState } from 'react'

export default function ProductModal({ item, backendUrl, onClose, onSelect, isSelected }) {
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchDetail = async () => {
      try {
        const res = await fetch(`${backendUrl}/items/${item.item_id}`)
        if (!res.ok) throw new Error()
        setDetail(await res.json())
      } catch {
        setDetail({ ...item, reviews: [] })
      } finally {
        setLoading(false)
      }
    }
    fetchDetail()
  }, [item.item_id, backendUrl])

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>✕</button>

        {loading ? (
          <div className="modal-loading">
            <div className="spinner" />
            <p>Đang tải thông tin sản phẩm...</p>
          </div>
        ) : (
          <>
            <div className="modal-header">
              <h3 className="modal-title">{detail?.title || item.title}</h3>
              <div className="modal-meta">
                <span className="modal-id">ID: {item.item_id}</span>
              </div>
            </div>

            <div className="modal-context">
              <h4>📝 Mô tả sản phẩm</h4>
              <p>{detail?.context || item.context || 'Chưa có mô tả.'}</p>
            </div>

            {/* Reviews */}
            {detail?.reviews?.length > 0 && (
              <div className="modal-reviews">
                <h4>⭐ Đánh giá từ người dùng</h4>
                <div className="review-list">
                  {detail.reviews.map((rev, i) => (
                    <div key={i} className="review-item">
                      <div className="review-stars">
                        {'★'.repeat(Math.round(rev.rating))}{'☆'.repeat(5 - Math.round(rev.rating))}
                        <span className="review-rating"> {rev.rating}/5</span>
                      </div>
                      <p className="review-text">{rev.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="modal-actions">
              <button className="btn-secondary" onClick={onClose}>Đóng</button>
              <button
                className={`btn-primary ${isSelected ? 'selected' : ''}`}
                onClick={onSelect}
                disabled={isSelected}
              >
                {isSelected ? '✓ Đã quan tâm' : '🛍️ Quan tâm sản phẩm này'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
