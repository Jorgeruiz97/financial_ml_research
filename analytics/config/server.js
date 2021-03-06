module.exports = ({ env }) => ({
  host: env('HOST', '0.0.0.0'),
  port: env.int('PORT', 1337),
  admin: {
    url: '/dashboard',
    auth: {
      secret: env('ADMIN_JWT_SECRET', '8faea0ed6eef2fffd3528e37433e0e68'),
    },
  },
});
