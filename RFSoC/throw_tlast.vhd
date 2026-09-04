library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity throw_tlast is
    generic (
        TDATA_WIDTH : integer := 128
    );
    port (
        clk              : in  STD_LOGIC;
        reset            : in  STD_LOGIC;
        packet_count_in  : in  STD_LOGIC_VECTOR(19 downto 0);
        s_axis_tdata     : in  STD_LOGIC_VECTOR(TDATA_WIDTH-1 downto 0);
        s_axis_tvalid    : in  STD_LOGIC;
        s_axis_tready    : out STD_LOGIC;
        m_axis_tdata     : out STD_LOGIC_VECTOR(TDATA_WIDTH-1 downto 0);
        m_axis_tvalid    : out STD_LOGIC;
        m_axis_tready    : in  STD_LOGIC;
        m_axis_tlast     : out STD_LOGIC
    );
end throw_tlast;

architecture Behavioral of throw_tlast is
    signal counter          : signed(19 downto 0) := (others => '0');
    signal packet_limit     : signed(19 downto 0) := (others => '0');
    signal packet_limit_prev: signed(19 downto 0) := (others => '0');
    signal limit_changed    : STD_LOGIC := '0';
begin

    s_axis_tready <= m_axis_tready;

    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '0' then
                m_axis_tdata      <= (others => '0');
                m_axis_tvalid     <= '0';
                m_axis_tlast      <= '0';
                counter           <= (others => '0');
                packet_limit      <= (others => '0');
                packet_limit_prev <= (others => '0');
                limit_changed     <= '0';
            else
                -- GPIO-Wert eintakten
                packet_limit      <= signed(packet_count_in);
                packet_limit_prev <= packet_limit;

                -- Änderungserkennung
                if packet_limit /= packet_limit_prev then
                    limit_changed <= '1';
                else
                    limit_changed <= '0';
                end if;

                -- Daten durchregistern
                m_axis_tdata  <= s_axis_tdata;
                m_axis_tvalid <= s_axis_tvalid;

                -- Zähler und TLAST
                if m_axis_tready = '1' and s_axis_tvalid = '1' then
                    if limit_changed = '1' then
                        m_axis_tlast <= '1';
                        counter      <= (others => '0');
                    elsif packet_limit > 1 and counter >= packet_limit - 1 then
                        m_axis_tlast <= '1';
                        counter      <= (others => '0');
                    else
                        m_axis_tlast <= '0';
                        counter      <= counter + 1;
                    end if;
                end if;
            end if;
        end if;
    end process;

end Behavioral;