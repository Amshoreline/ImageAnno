import Vue from 'vue';
import Router from 'vue-router';
import MessageBox from '../components/MessageBox.vue'
import Test from '../components/Test.vue'

Vue.use(Router);

export default new Router({
  routes: [
    {
      path: '/',
      name: 'Test',
      component: Test
    },
    {
      path: '/about',
      name: 'Message',
      component: MessageBox
    }
  ]
});
