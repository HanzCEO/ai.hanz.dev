import { Route, Routes } from 'react-router'

import RouteHead from '@/components/RouteHead'
import SiteLayout from '@/components/layout/SiteLayout'
import { KV_CACHE_PATH, REAP_PATH } from '@/lib/seo'
import Home from '@/routes/Home'
import KvCacheCalculator from '@/routes/KvCacheCalculator'
import NotFound from '@/routes/NotFound'
import ReapCostCalculator from '@/routes/ReapCostCalculator'

export default function App() {
  return (
    <SiteLayout>
      <RouteHead />
      <Routes>
        <Route path="/" element={<Home />} />
        {/*
          The tool routes read their paths from ROUTE_META so the router and the
          canonical URLs cannot disagree about where a page lives. React Router
          treats the trailing slash as insignificant while matching, so both
          request forms still land here.
        */}
        <Route path={KV_CACHE_PATH} element={<KvCacheCalculator />} />
        <Route path={REAP_PATH} element={<ReapCostCalculator />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </SiteLayout>
  )
}
