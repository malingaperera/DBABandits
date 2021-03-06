CREATE NONCLUSTERED INDEX IXN_LINEITEM_l_shipdate_l_suppkey_l_orderkey_l_ex_l_di ON LINEITEM (L_SHIPDATE, L_SUPPKEY, L_ORDERKEY) INCLUDE (L_EXTENDEDPRICE, L_DISCOUNT);


CREATE NONCLUSTERED INDEX IXN_LINEITEM_l_shipdate_l_qu_l_ex_l_ta_l_di_l_re_l_li ON LINEITEM (L_SHIPDATE) INCLUDE (L_QUANTITY, L_EXTENDEDPRICE, L_TAX, L_DISCOUNT, L_RETURNFLAG, L_LINESTATUS);


CREATE NONCLUSTERED INDEX IX_LINEITEM_l_orderkey_l_receiptdate_l_commitdate_l_suppkey ON LINEITEM (L_ORDERKEY, L_RECEIPTDATE, L_COMMITDATE, L_SUPPKEY);


CREATE NONCLUSTERED INDEX IXN_LINEITEM_l_orderkey_l_partkey_l_suppkey_l_ex_l_di_l_qu ON LINEITEM (L_ORDERKEY, L_PARTKEY, L_SUPPKEY) INCLUDE (L_EXTENDEDPRICE, L_DISCOUNT, L_QUANTITY);


CREATE NONCLUSTERED INDEX IX_LINEITEM_l_shipmode_l_receiptdate_l_orderkey_l_shipdate_l_commitdate ON LINEITEM (L_SHIPMODE, L_RECEIPTDATE, L_ORDERKEY, L_SHIPDATE, L_COMMITDATE);


CREATE NONCLUSTERED INDEX IXN_ORDERS_o_orderdate_o_custkey_o_orderkey_o_sh ON ORDERS (O_ORDERDATE, O_CUSTKEY, O_ORDERKEY) INCLUDE (O_SHIPPRIORITY);


CREATE NONCLUSTERED INDEX IX_ORDERS_o_orderstatus_o_orderkey ON ORDERS (O_ORDERSTATUS, O_ORDERKEY);


CREATE NONCLUSTERED INDEX IXN_ORDERS_o_orderdate_o_orderkey_o_or ON ORDERS (O_ORDERKEY) INCLUDE (O_ORDERDATE);


CREATE NONCLUSTERED INDEX IXN_PARTSUPP_ps_suppkey_ps_partkey_ps_s ON PARTSUPP (PS_SUPPKEY, PS_PARTKEY) INCLUDE (PS_SUPPLYCOST);


CREATE NONCLUSTERED INDEX IX_CUSTOMER_c_mktsegment ON CUSTOMER (C_MKTSEGMENT);


CREATE NONCLUSTERED INDEX IX_CUSTOMER_c_nationkey_c_custkey ON CUSTOMER (C_NATIONKEY, C_CUSTKEY);


CREATE NONCLUSTERED INDEX IX_CUSTOMER_c_acctbal_c_phone ON CUSTOMER (C_ACCTBAL, C_PHONE);


CREATE NONCLUSTERED INDEX IX_PART_p_type_p_partkey ON PART (P_TYPE, P_PARTKEY);


CREATE NONCLUSTERED INDEX IX_PART_p_container_p_size_p_partkey_p_brand ON PART (P_SIZE, P_PARTKEY, P_BRAND);


CREATE NONCLUSTERED INDEX IX_PART_p_name ON PART (P_NAME);


CREATE NONCLUSTERED INDEX IX_PART_p_brand_p_type_p_size_p_partkey ON PART (P_BRAND, P_TYPE, P_SIZE, P_PARTKEY);


CREATE NONCLUSTERED INDEX IXN_PART_p_size_p_partkey_p_type_p_mf ON PART (P_PARTKEY, P_TYPE) INCLUDE (P_MFGR);


CREATE NONCLUSTERED INDEX IX_REGION_r_name ON REGION (R_NAME);


CREATE NONCLUSTERED INDEX IX_NATION_n_name_n_nationkey ON NATION (N_NAME, N_NATIONKEY);


CREATE NONCLUSTERED INDEX IXN_SUPPLIER_s_nationkey_s_suppkey_s_na_s_ph_s_co_s_ac_s_ad ON SUPPLIER (S_NATIONKEY, S_SUPPKEY) INCLUDE (S_NAME, S_PHONE, S_COMMENT, S_ACCTBAL, S_ADDRESS);