import { Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import { HealthProvider } from './health'
import Home from './pages/Home'
import Studio from './pages/Studio'
import Architecture from './pages/Architecture'
import Gallery from './pages/Gallery'

export default function App() {
  return (
    <HealthProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="studio" element={<Studio />} />
          <Route path="architecture" element={<Architecture />} />
          <Route path="gallery" element={<Gallery />} />
          <Route path="*" element={<Home />} />
        </Route>
      </Routes>
    </HealthProvider>
  )
}
