import { useState, useEffect } from 'react'
import ProductModal from './ProductModal'

// Một vài danh mục tiêu biểu để hiển thị trong cửa hàng
const DEMO_CATEGORIES = [
  { id: 'electronics', label: '📱 Điện tử', icon: '📱' },
  { id: 'books',       label: '📚 Sách',    icon: '📚' },
  { id: 'clothing',    label: '👕 Thời trang', icon: '👕' },
  { id: 'home',        label: '🏠 Nhà cửa', icon: '🏠' },
  { id: 'sports',      label: '⚽ Thể thao', icon: '⚽' },
]

// Màu gradient cho từng danh mục
const CATEGORY_COLORS = {
  electronics: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
  books:       'linear-gradient(135deg, #2d1b69 0%, #11998e 100%)',
  clothing:    'linear-gradient(135deg, #8e0e00 0%, #1f1c18 100%)',
  home:        'linear-gradient(135deg, #134e5e 0%, #71b280 100%)',
  sports:      'linear-gradient(135deg, #232526 0%, #414345 100%)',
}

export default function ProductGrid({ onItemClick, clickedItems, backendUrl }) {
  const [products, setProducts]         = useState([])
  const [loading, setLoading]           = useState(true)
  const [selectedCategory, setSelectedCategory] = useState('electronics')
  const [modalItem, setModalItem]       = useState(null)

  // Tải danh sách sản phẩm mẫu từ Qdrant thông qua Backend
  useEffect(() => {
    const fetchProducts = async () => {
      setLoading(true)
      try {
        const res = await fetch(`${backendUrl}/products?category=${selectedCategory}&limit=100`)
        if (!res.ok) throw new Error('Không thể tải sản phẩm')
        const data = await res.json()
        setProducts(data.products || [])
      } catch {
        // Nếu backend chưa sẵn sàng, dùng dữ liệu mẫu
        setProducts(generateMockProducts(selectedCategory))
      } finally {
        setLoading(false)
      }
    }
    fetchProducts()
  }, [selectedCategory, backendUrl])

  const isClicked = (item) => clickedItems.some(c => c.item_id === item.item_id)

  return (
    <div className="product-grid-wrapper">
      {/* Thanh danh mục */}
      <div className="category-tabs">
        {DEMO_CATEGORIES.map(cat => (
          <button
            key={cat.id}
            className={`category-tab ${selectedCategory === cat.id ? 'active' : ''}`}
            onClick={() => setSelectedCategory(cat.id)}
          >
            {cat.icon} {cat.label.split(' ').slice(1).join(' ')}
          </button>
        ))}
      </div>

      {/* Grid sản phẩm */}
      {loading ? (
        <div className="loading-grid">
          {Array.from({ length: 12 }).map((_, i) => (
            <div key={i} className="product-skeleton" />
          ))}
        </div>
      ) : (
        <div className="product-grid">
          {products.map((item) => (
            <div
              key={item.item_id}
              className={`product-card ${isClicked(item) ? 'selected' : ''}`}
              style={{ background: CATEGORY_COLORS[selectedCategory] }}
            >
              <div className="product-card-inner">
                {item.image ? (
                  <img src={item.image} alt={item.title} className="product-image" />
                ) : (
                  <div className="product-icon">{DEMO_CATEGORIES.find(c => c.id === selectedCategory)?.icon}</div>
                )}
                <div className="product-info">
                  <div className="product-title">{item.title}</div>
                  {item.rating && (
                    <div className="product-rating">
                      <span className="stars">{'★'.repeat(Math.round(item.rating))}{'☆'.repeat(5-Math.round(item.rating))}</span>
                      <span className="rating-score"> {item.rating} ({item.review_count})</span>
                    </div>
                  )}
                  {item.context && (
                    <div className="product-review">"{item.context.substring(0, 80)}..."</div>
                  )}
                </div>
              </div>

              <div className="product-actions">
                <button
                  className="btn-detail"
                  onClick={() => setModalItem(item)}
                >
                  Chi tiết
                </button>
                <button
                  className={`btn-buy ${isClicked(item) ? 'bought' : ''}`}
                  onClick={() => onItemClick(item)}
                  disabled={isClicked(item)}
                >
                  {isClicked(item) ? '✓ Đã chọn' : 'Quan tâm →'}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Modal xem chi tiết */}
      {modalItem && (
        <ProductModal
          item={modalItem}
          backendUrl={backendUrl}
          onClose={() => setModalItem(null)}
          onSelect={() => { onItemClick(modalItem); setModalItem(null) }}
          isSelected={isClicked(modalItem)}
        />
      )}
    </div>
  )
}

// Dữ liệu mẫu khi backend chưa sẵn sàng
function generateMockProducts(category) {
  const titles = {
    electronics: ['Tai nghe Sony WH-1000XM5', 'Chuột Logitech MX Master 3', 'Bàn phím cơ Keychron K2', 'Webcam Logitech C920', 'SSD Samsung 1TB', 'Màn hình LG 27" 4K', 'Laptop MacBook Air M2', 'iPad Pro 12.9"', 'Apple Watch Series 9', 'AirPods Pro 2', 'Loa JBL Charge 5', 'Ổ cứng di động WD 2TB', 'Router WiFi 6 TP-Link', 'Bàn phím HHKB Pro', 'Micro Blue Yeti', 'Card màn hình RTX 4070'],
    books:       ['Nhà Giả Kim - Paulo Coelho', 'Đắc Nhân Tâm', 'Sapiens - Lược Sử Loài Người', 'The Alchemist', 'Deep Work', 'Atomic Habits', 'Thinking Fast and Slow', 'Rich Dad Poor Dad', '48 Laws of Power', 'The Psychology of Money', 'Zero to One', 'The Lean Startup', 'Clean Code', 'Design Patterns', 'The Pragmatic Programmer', 'Structure and Interpretation'],
    clothing:    ['Áo polo Ralph Lauren', 'Quần jean Levi\'s 501', 'Áo hoodie Nike Tech Fleece', 'Giày Adidas Ultraboost', 'Túi xách Michael Kors', 'Đồng hồ Seiko', 'Kính mắt Ray-Ban Aviator', 'Áo khoác The North Face', 'Sneakers New Balance 574', 'Áo thun Uniqlo', 'Quần âu Zara', 'Giày Dr. Martens', 'Mũ bucket Columbia', 'Balo Herschel', 'Ví da Coach', 'Thắt lưng Fossil'],
    home:        ['Nồi cơm điện Zojirushi', 'Máy lọc không khí Xiaomi', 'Robot hút bụi Roomba', 'Nồi chiên không dầu Philips', 'Đèn bàn Xiaomi Mi', 'Máy pha cà phê Nespresso', 'Bộ chăn ga Tencel', 'Giá sách IKEA', 'Đèn ngủ Philips Hue', 'Máy xay sinh tố Vitamix', 'Nồi áp suất Instant Pot', 'Máy lọc nước RO', 'Tủ giày thông minh', 'Gương LED phòng tắm', 'Máy sấy quần áo', 'Lò vi sóng Panasonic'],
    sports:      ['Giày chạy bộ Nike Pegasus', 'Máy tập đa năng Bowflex', 'Dây kháng lực set 5 cái', 'Thảm yoga TPE 6mm', 'Bình nước Hydro Flask', 'Áo tập Under Armour', 'Bóng rổ Spalding', 'Gậy tennis Wilson', 'Bao tay boxing Everlast', 'Xe đạp tập Peloton', 'Quả tạ điều chỉnh 20kg', 'Băng quấn cổ tay', 'Kính bơi Speedo', 'Áo vest bơi TYR', 'Túi gym Nike', 'Máy đo nhịp tim Garmin'],
  }
  return (titles[category] || titles.electronics).map((title, i) => ({
    item_id: `mock-${category}-${i}`,
    title,
    score: 0.85 - i * 0.01,
    context: `Sản phẩm ${title} được đánh giá cao bởi người dùng Amazon.`,
  }))
}
