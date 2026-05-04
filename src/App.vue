<template>
  <Header />
  <div class="image-wrap">
    <img src="@/assets/images/sea.png" alt="feature image" class="w-full" />
  </div>
  <div class="max-w-5xl mx-auto px-6 py-10 flex flex-col md:flex-row gap-10">
    <Sidebar />
    <main class="flex-1 min-w-0">
      <router-view />
    </main>
  </div>
  <Footer />
</template>

<script setup>
import Header from '@/components/Header.vue'
import Footer from '@/components/Footer.vue'
import Sidebar from '@/components/Sidebar.vue'
</script>
