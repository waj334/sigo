#ifndef LWIPOPTS_H
#define LWIPOPTS_H

/* Bare-metal, no OS */
#define NO_SYS                  1

/* Disable sequential/socket APIs (require NO_SYS=0) */
#define LWIP_SOCKET             0
#define LWIP_NETCONN            0

/* Memory settings */
#define MEM_ALIGNMENT           4
#define MEM_SIZE                (16 * 1024)
#define MEMP_NUM_PBUF           16
#define PBUF_POOL_SIZE          16

/* Core protocols */
#define LWIP_ARP                1
#define LWIP_ICMP               1
#define LWIP_UDP                1
#define LWIP_TCP                1
#define LWIP_DHCP               1

/* IPv4 only */
#define LWIP_IPV4               1
#define LWIP_IPV6               0

/* TCP tuning */
#define TCP_MSS                 1460
#define TCP_WND                 (4 * TCP_MSS)
#define TCP_SND_BUF             (4 * TCP_MSS)

/* Debugging */
#define LWIP_DEBUG              0
#define LWIP_DBG_MIN_LEVEL      LWIP_DBG_LEVEL_ALL

#define ETHARP_DEBUG            LWIP_DBG_OFF
#define NETIF_DEBUG             LWIP_DBG_OFF
#define PBUF_DEBUG              LWIP_DBG_OFF
#define IP_DEBUG                LWIP_DBG_OFF
#define DHCP_DEBUG              LWIP_DBG_OFF
#define TCP_DEBUG               LWIP_DBG_OFF
#define UDP_DEBUG               LWIP_DBG_OFF

#endif /* LWIPOPTS_H */
