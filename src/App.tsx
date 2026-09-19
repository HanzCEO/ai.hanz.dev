import { Route, Routes } from 'react-router'

import RouteHead from '@/components/RouteHead'
import SiteLayout from '@/components/layout/SiteLayout'
import Home from '@/routes/Home'
import KvCacheCalculator from '@/routes/KvCacheCalculator'
import NotFound from '@/routes/NotFound'

export default function App() {
  return (
    <SiteLayout>
      <RouteHead />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/tools/kv-cache-calculator" element={<KvCacheCalculator />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </SiteLayout>
  )
}
