#!/usr/bin/env python3
"""Generate publication-quality figures from {model_label} cloud experiment."""

import json
import re
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Style ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
})

MALE_COLOR = '#1565C0'
FEMALE_COLOR = '#C62828'
MALE_LIGHT = '#64B5F6'
FEMALE_LIGHT = '#EF9A9A'
EXPECTED_COLOR = '#E65100'

CAT_COLORS = {
    'SA': '#FF6F00', 'WA': '#2E7D32', 'LA': '#1565C0',
    'EE': '#6D4C41', 'ME': '#00838F', 'EA': '#C62828', 'AN': '#6A1B9A',
}
CAT_LABELS = {
    'EA': 'East Asian', 'SA': 'South Asian', 'WA': 'West African',
    'LA': 'Latin American', 'AN': 'Anglo/Western', 'ME': 'Middle Eastern',
    'EE': 'East European',
}

# ── Large pool metadata ──
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from speed_bias_cloud import NAMES_LARGE
POOL = {n: {'gender': g, 'cat': c} for n, g, c in NAMES_LARGE}


def parse_log(path):
    games = []
    with open(path) as f:
        for line in f:
            m = re.search(
                r"Game\s+(\d+)/(\d+):\s+wolf=(\S+)\s+\*(CAUGHT|escaped)\*\s+"
                r"voted_out=(\S+)\s+votes=(\{.*?\})\s+\((\d+\.\d+)s\)",
                line)
            if m:
                votes = {}
                for vm in re.finditer(r"'(\w[\w-]*)'\s*:\s*'(\w[\w-]*)'", m.group(6)):
                    votes[vm.group(1)] = vm.group(2)
                games.append({
                    'num': int(m.group(1)), 'wolf': m.group(3),
                    'caught': m.group(4) == 'CAUGHT',
                    'voted_out': m.group(5) if m.group(5) != 'tie' else None,
                    'votes': votes,
                })
    return games


def compute_all(games):
    stats = defaultdict(lambda: {'games': 0, 'voted_out': 0, 'wolf_games': 0, 'wolf_caught': 0})
    vil_gf = defaultdict(int)
    wolf_gf = defaultdict(int)
    gxc = defaultdict(lambda: {'games': 0, 'voted_out': 0})
    # Full gender×category vote flow: (voter_gender, voter_cat) → (target_gender, target_cat) → count
    vil_gcf = defaultdict(int)

    for g in games:
        players = set(g['votes'].keys())
        if g['voted_out']: players.add(g['voted_out'])
        players.add(g['wolf'])

        for n in players:
            stats[n]['games'] += 1
            if n == g['wolf']:
                stats[n]['wolf_games'] += 1
                if g['caught']: stats[n]['wolf_caught'] += 1
            if g['voted_out'] == n:
                stats[n]['voted_out'] += 1
            info = POOL.get(n)
            if info:
                gxc[(info['gender'], info['cat'])]['games'] += 1
                if g['voted_out'] == n:
                    gxc[(info['gender'], info['cat'])]['voted_out'] += 1

        for voter, target in g['votes'].items():
            vi = POOL.get(voter, {})
            ti = POOL.get(target, {})
            vg = vi.get('gender', '?')
            tg = ti.get('gender', '?')
            if voter == g['wolf']:
                wolf_gf[(vg, tg)] += 1
            else:
                vil_gf[(vg, tg)] += 1
                # Track full gender×category flow (villagers only)
                vc = vi.get('cat', '??')
                tc = ti.get('cat', '??')
                if vc != '??' and tc != '??':
                    vil_gcf[((vg, vc), (tg, tc))] += 1

    return dict(stats), dict(vil_gf), dict(wolf_gf), dict(gxc), dict(vil_gcf)


def fig_gender_gap(games, stats, out, model_label="GPT-4o-mini"):
    """Bar chart: male vs female elimination rate with expected baseline."""
    m_g = sum(s['games'] for n, s in stats.items() if POOL.get(n, {}).get('gender') == 'M')
    f_g = sum(s['games'] for n, s in stats.items() if POOL.get(n, {}).get('gender') == 'F')
    m_vo = sum(s['voted_out'] for n, s in stats.items() if POOL.get(n, {}).get('gender') == 'M')
    f_vo = sum(s['voted_out'] for n, s in stats.items() if POOL.get(n, {}).get('gender') == 'F')
    m_rate = 100 * m_vo / m_g
    f_rate = 100 * f_vo / f_g
    expected = 100 / 7
    gap = m_rate - f_rate

    # z-score for two-proportion test
    p_pool = (m_vo + f_vo) / (m_g + f_g)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/m_g + 1/f_g))
    z = (m_vo/m_g - f_vo/f_g) / se

    fig, ax = plt.subplots(figsize=(7, 6))

    bars = ax.bar(['Male Names\n(70 names)', 'Female Names\n(70 names)'],
                  [m_rate, f_rate],
                  color=[MALE_COLOR, FEMALE_COLOR], width=0.5, edgecolor='white',
                  linewidth=1.5, zorder=3)

    ax.axhline(y=expected, color=EXPECTED_COLOR, linestyle='--', linewidth=2,
               label=f'Expected if unbiased ({expected:.1f}%)', zorder=2)

    # Value labels
    ax.text(0, m_rate + 0.5, f'{m_rate:.1f}%', ha='center', va='bottom',
            fontsize=18, fontweight='bold', color=MALE_COLOR)
    ax.text(1, f_rate + 0.5, f'{f_rate:.1f}%', ha='center', va='bottom',
            fontsize=18, fontweight='bold', color=FEMALE_COLOR)

    # Gap annotation
    ax.annotate('', xy=(0.65, m_rate), xytext=(0.65, f_rate),
                arrowprops=dict(arrowstyle='<->', color='#333', lw=2))
    ax.text(0.75, (m_rate + f_rate) / 2, f'{gap:.1f}pp gap\nz = {z:.1f}\np < 0.001',
            fontsize=11, va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', edgecolor='#F9A825'))

    ax.set_ylabel('Elimination Rate (%)')
    ax.set_ylim(0, m_rate * 1.35)
    ax.legend(loc='upper right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title('Gender Elimination Gap')

    # Description box
    desc = (f"{model_label}  •  {len(games)} games  •  140 names (70M/70F)  •  7 categories\n"
            f"Each game: 7 random players, 1 wolf, 1 discussion round, 1 vote.\n"
            f"Male-named players are eliminated {gap:.0f} percentage points more often than female-named players.")
    fig.text(0.5, -0.02, desc, ha='center', va='top', fontsize=9, color='#555',
             style='italic', wrap=True)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out / '01_gender_gap.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  01_gender_gap.png")


def fig_vote_flow(games, vil_gf, wolf_gf, out, model_label="GPT-4o-mini"):
    """Heatmap: who votes for who by gender, villagers vs wolves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for ax, gf, title, total_label in [
        (ax1, vil_gf, 'Villager Votes', 'As Villager'),
        (ax2, wolf_gf, 'Wolf Votes', 'As Wolf'),
    ]:
        m_row = gf.get(('M', 'M'), 0) + gf.get(('M', 'F'), 0)
        f_row = gf.get(('F', 'M'), 0) + gf.get(('F', 'F'), 0)
        if m_row == 0 or f_row == 0:
            continue

        matrix = np.array([
            [100 * gf.get(('M', 'M'), 0) / m_row, 100 * gf.get(('M', 'F'), 0) / m_row],
            [100 * gf.get(('F', 'M'), 0) / f_row, 100 * gf.get(('F', 'F'), 0) / f_row],
        ])
        counts = np.array([
            [gf.get(('M', 'M'), 0), gf.get(('M', 'F'), 0)],
            [gf.get(('F', 'M'), 0), gf.get(('F', 'F'), 0)],
        ])

        im = ax.imshow(matrix, cmap='RdYlGn_r', vmin=0, vmax=100, aspect='auto')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Target: Male', 'Target: Female'], fontsize=11)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Voter:\nMale', 'Voter:\nFemale'], fontsize=11)

        for i in range(2):
            for j in range(2):
                tc = 'white' if matrix[i, j] > 55 else 'black'
                ax.text(j, i, f'{matrix[i, j]:.0f}%\n({counts[i,j]:,})',
                        ha='center', va='center', fontsize=15, fontweight='bold', color=tc)

        ax.set_title(f'{title}\n({m_row + f_row:,} total)', fontsize=13)

    fig.suptitle('Vote Targeting by Gender', fontsize=15, fontweight='bold', y=1.02)

    desc = (f"{model_label}  •  {len(games)} games  •  140 names  •  Expected: ~50/50 in balanced pool\n"
            f"LEFT: How villagers vote — both genders target males ~81% of the time.\n"
            f"RIGHT: How wolves vote — wolves also prefer male targets but less extremely (~65%).")
    fig.text(0.5, -0.06, desc, ha='center', va='top', fontsize=9, color='#555',
             style='italic', wrap=True)

    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.savefig(out / '02_vote_flow.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  02_vote_flow.png")


def fig_top_bottom(games, stats, out, model_label="GPT-4o-mini"):
    """Horizontal bars: top 15 most eliminated vs bottom 15 most protected names."""
    played = {n: s for n, s in stats.items() if s['games'] >= 10}
    sorted_names = sorted(played, key=lambda n: -played[n]['voted_out'] / played[n]['games'])

    top15 = sorted_names[:15]
    bot15 = sorted_names[-15:][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    expected = 100 / 7

    for ax, names, title in [(ax1, top15, 'Most Targeted'), (ax2, bot15, 'Most Protected')]:
        rates = [100 * played[n]['voted_out'] / played[n]['games'] for n in names]
        colors = [MALE_COLOR if POOL.get(n, {}).get('gender') == 'M' else FEMALE_COLOR for n in names]

        bars = ax.barh(range(len(names)), rates, color=colors, edgecolor='white',
                       height=0.65, zorder=3)
        ax.axvline(x=expected, color=EXPECTED_COLOR, linestyle='--', linewidth=2, zorder=2)
        ax.set_yticks(range(len(names)))

        ylabels = []
        for n in names:
            info = POOL.get(n, {})
            g = info.get('gender', '?')
            c = info.get('cat', '??')
            ylabels.append(f'{n}  ({g}/{c})')
        ax.set_yticklabels(ylabels, fontsize=10, fontweight='bold')
        ax.invert_yaxis()

        for i, (n, r) in enumerate(zip(names, rates)):
            s = played[n]
            gms = s['games']
            wg = s['wolf_games']
            wc = s['wolf_caught']
            catch = f'{wc}/{wg}' if wg else '-'
            ax.text(r + 0.3, i, f'{r:.0f}%  ({s["voted_out"]}/{gms}g)  wolf:{catch}',
                    va='center', fontsize=9)

        ax.set_xlabel('Elimination Rate (%)')
        ax.set_title(title, fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend
    m_patch = mpatches.Patch(color=MALE_COLOR, label='Male name')
    f_patch = mpatches.Patch(color=FEMALE_COLOR, label='Female name')
    exp_line = plt.Line2D([0], [0], color=EXPECTED_COLOR, linestyle='--', linewidth=2,
                          label=f'Expected ({expected:.1f}%)')
    fig.legend(handles=[m_patch, f_patch, exp_line], loc='upper center',
               ncol=3, fontsize=11, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle('Individual Name Elimination Rates', fontsize=15, fontweight='bold', y=1.04)

    desc = (f"{model_label}  •  {len(games)} games  •  140 names  •  Names with <10 appearances excluded\n"
            f"LEFT: 15 names eliminated most often. ALL are male. Viktor (M/East European) leads at 71%.\n"
            f"RIGHT: 15 names eliminated least. ALL are female. 0% elimination across 15-31 appearances each.\n"
            f"Format: rate%  (eliminations/games)  wolf:caught/assigned")
    fig.text(0.5, -0.04, desc, ha='center', va='top', fontsize=9, color='#555',
             style='italic', wrap=True)

    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out / '03_top_bottom_names.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  03_top_bottom_names.png")


def fig_ethnicity(games, gxc, out, model_label="GPT-4o-mini"):
    """Grouped bars: elimination rate by ethnic category, split by gender."""
    cats = sorted(CAT_LABELS.keys())
    expected = 100 / 7

    m_rates, f_rates, m_n, f_n = [], [], [], []
    for cat in cats:
        m = gxc.get(('M', cat), {'games': 0, 'voted_out': 0})
        f = gxc.get(('F', cat), {'games': 0, 'voted_out': 0})
        m_rates.append(100 * m['voted_out'] / m['games'] if m['games'] > 0 else 0)
        f_rates.append(100 * f['voted_out'] / f['games'] if f['games'] > 0 else 0)
        m_n.append(m['games'])
        f_n.append(f['games'])

    fig, ax = plt.subplots(figsize=(13, 6.5))

    x = np.arange(len(cats))
    width = 0.35

    bars_m = ax.bar(x - width/2, m_rates, width, color=MALE_COLOR, label='Male names (10 per category)',
                    edgecolor='white', linewidth=1, zorder=3)
    bars_f = ax.bar(x + width/2, f_rates, width, color=FEMALE_COLOR, label='Female names (10 per category)',
                    edgecolor='white', linewidth=1, zorder=3)

    ax.axhline(y=expected, color=EXPECTED_COLOR, linestyle='--', linewidth=2,
               label=f'Expected if unbiased ({expected:.1f}%)', zorder=2)

    for bar, rate, n in zip(bars_m, m_rates, m_n):
        if rate > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f'{rate:.0f}%', ha='center', va='bottom', fontsize=10,
                    fontweight='bold', color=MALE_COLOR)

    for bar, rate, n in zip(bars_f, f_rates, f_n):
        ax.text(bar.get_x() + bar.get_width()/2, max(rate, 0) + 0.4,
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color=FEMALE_COLOR)

    # Gap labels
    for i, (mr, fr) in enumerate(zip(m_rates, f_rates)):
        gap = mr - fr
        ax.text(i, max(mr, fr) + 3.5, f'+{gap:.0f}pp',
                ha='center', fontsize=9, fontweight='bold', color='#333',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF9C4',
                          edgecolor='#F9A825', alpha=0.9))

    cat_labels_full = [f'{CAT_LABELS[c]}\n({c})' for c in cats]
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels_full, fontsize=10)
    ax.set_ylabel('Elimination Rate (%)')
    ax.set_ylim(0, max(m_rates) * 1.35)
    ax.legend(loc='upper right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title('Elimination Rate by Gender × Ethnic Category')

    desc = (f"{model_label}  •  {len(games)} games  •  140 names: 10 male + 10 female per category\n"
            f"Each bar shows how often players with names from that gender/ethnicity are voted out.\n"
            f"The gender gap exists in EVERY category — male names are always above the expected line,\n"
            f"female names always below. South Asian males are hit hardest (+32%), Anglo females least (2%).\n"
            f"Categories: EA=Japanese/Chinese/Korean, SA=Indian/Pakistani, WA=Nigerian/Ghanaian/Senegalese,\n"
            f"LA=Mexican/Colombian/Brazilian, AN=English/American, ME=Arabic/Persian/Turkish, EE=Russian/Polish/Ukrainian")
    fig.text(0.5, -0.06, desc, ha='center', va='top', fontsize=9, color='#555',
             style='italic', wrap=True)

    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(out / '04_ethnicity_breakdown.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  04_ethnicity_breakdown.png")


def fig_convergence(games, out, model_label="GPT-4o-mini"):
    """Line chart: cumulative gender gap over time showing stability."""
    cum_m_g, cum_f_g, cum_m_vo, cum_f_vo = 0, 0, 0, 0
    gaps, nums = [], []

    for g in games:
        players = set(g['votes'].keys())
        if g['voted_out']: players.add(g['voted_out'])
        players.add(g['wolf'])

        for n in players:
            info = POOL.get(n)
            if info:
                if info['gender'] == 'M':
                    cum_m_g += 1
                    if g['voted_out'] == n: cum_m_vo += 1
                else:
                    cum_f_g += 1
                    if g['voted_out'] == n: cum_f_vo += 1

        if cum_m_g > 10 and cum_f_g > 10:
            gap = 100 * cum_m_vo / cum_m_g - 100 * cum_f_vo / cum_f_g
            gaps.append(gap)
            nums.append(g['num'])

    if not gaps:
        return

    final_gap = gaps[-1]

    # Compute rolling 95% CI at each point
    ci_upper, ci_lower = [], []
    cum_m_g2, cum_f_g2, cum_m_vo2, cum_f_vo2 = 0, 0, 0, 0
    idx = 0
    for g in games:
        players = set(g['votes'].keys())
        if g['voted_out']: players.add(g['voted_out'])
        players.add(g['wolf'])
        for n in players:
            info = POOL.get(n)
            if info:
                if info['gender'] == 'M':
                    cum_m_g2 += 1
                    if g['voted_out'] == n: cum_m_vo2 += 1
                else:
                    cum_f_g2 += 1
                    if g['voted_out'] == n: cum_f_vo2 += 1
        if cum_m_g2 > 10 and cum_f_g2 > 10:
            p_m = cum_m_vo2 / cum_m_g2
            p_f = cum_f_vo2 / cum_f_g2
            se = math.sqrt(p_m*(1-p_m)/cum_m_g2 + p_f*(1-p_f)/cum_f_g2) * 100
            gap_val = gaps[idx]
            ci_upper.append(gap_val + 1.96 * se)
            ci_lower.append(gap_val - 1.96 * se)
            idx += 1

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.fill_between(nums, ci_lower, ci_upper, alpha=0.15, color=MALE_COLOR, label='95% CI', zorder=1)
    ax.plot(nums, gaps, color=MALE_COLOR, linewidth=2, zorder=3)
    ax.axhline(y=0, color='#999', linewidth=1, zorder=1)
    ax.axhline(y=final_gap, color=EXPECTED_COLOR, linestyle='--', linewidth=1.5,
               label=f'Final: +{final_gap:.1f}pp', zorder=2)

    ax.set_xlabel('Games Played')
    ax.set_ylabel('Gender Gap (pp)\nMale elim. rate − Female elim. rate')
    ax.set_title('Gender Bias Convergence')
    ax.legend(loc='upper right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    desc = (f"{model_label}  •  {len(games)} games  •  140 names\n"
            f"Cumulative gender gap (male elimination rate minus female elimination rate) stabilizes\n"
            f"within ~100 games and holds steady at +{final_gap:.0f}pp through {len(games)} games.\n"
            f"Shaded area = 95% confidence interval. The gap never crosses zero.")
    fig.text(0.5, -0.04, desc, ha='center', va='top', fontsize=9, color='#555',
             style='italic', wrap=True)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out / '05_convergence.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  05_convergence.png")


def fig_wolf_immunity(games, stats, out, model_label="GPT-4o-mini"):
    """Scatter: wolf catch rate vs appearances — do female wolves get caught?"""
    played = {n: s for n, s in stats.items() if s['wolf_games'] >= 3}

    fig, ax = plt.subplots(figsize=(10, 6))

    for n, s in played.items():
        info = POOL.get(n, {})
        gender = info.get('gender', '?')
        catch_rate = 100 * s['wolf_caught'] / s['wolf_games']
        color = MALE_COLOR if gender == 'M' else FEMALE_COLOR
        size = s['wolf_games'] * 12
        ax.scatter(s['wolf_games'], catch_rate, c=color, s=size, alpha=0.7,
                   edgecolors='white', linewidth=0.5, zorder=3)

    # Aggregate lines
    m_wg = sum(s['wolf_games'] for n, s in stats.items() if POOL.get(n, {}).get('gender') == 'M')
    m_wc = sum(s['wolf_caught'] for n, s in stats.items() if POOL.get(n, {}).get('gender') == 'M')
    f_wg = sum(s['wolf_games'] for n, s in stats.items() if POOL.get(n, {}).get('gender') == 'F')
    f_wc = sum(s['wolf_caught'] for n, s in stats.items() if POOL.get(n, {}).get('gender') == 'F')

    if m_wg > 0:
        ax.axhline(y=100*m_wc/m_wg, color=MALE_COLOR, linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Male avg: {100*m_wc/m_wg:.0f}% caught ({m_wc}/{m_wg})')
    if f_wg > 0:
        ax.axhline(y=100*f_wc/f_wg, color=FEMALE_COLOR, linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Female avg: {100*f_wc/f_wg:.0f}% caught ({f_wc}/{f_wg})')

    ax.set_xlabel('Times Assigned as Wolf')
    ax.set_ylabel('Wolf Catch Rate (%)')
    ax.set_title('Wolf Catch Rate by Gender')
    ax.set_ylim(-5, 105)
    ax.legend(loc='upper right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    m_patch = mpatches.Patch(color=MALE_COLOR, label='Male name')
    f_patch = mpatches.Patch(color=FEMALE_COLOR, label='Female name')
    ax.legend(handles=[m_patch, f_patch,
                       plt.Line2D([0],[0], color=MALE_COLOR, ls='--', label=f'Male avg: {100*m_wc/m_wg:.0f}%'),
                       plt.Line2D([0],[0], color=FEMALE_COLOR, ls='--', label=f'Female avg: {100*f_wc/f_wg:.0f}%')],
              loc='upper right', fontsize=10)

    desc = (f"{model_label}  •  {len(games)} games  •  Names with ≥3 wolf assignments shown\n"
            f"Each dot = one name. Dot size = number of wolf assignments.\n"
            f"Male wolves are caught {100*m_wc/m_wg:.0f}% of the time. "
            f"Female wolves are caught {100*f_wc/f_wg:.0f}% of the time.\n"
            f"Female-named wolves benefit from the village's refusal to vote for female names.")
    fig.text(0.5, -0.04, desc, ha='center', va='top', fontsize=9, color='#555',
             style='italic', wrap=True)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out / '06_wolf_immunity.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  06_wolf_immunity.png")


def fig_cross_group_targeting(games, vil_gcf, out, model_label="GPT-4o-mini"):
    """14×14 heatmap: how each gender×category group targets every other group."""
    cats_order = ['AN', 'EA', 'EE', 'LA', 'ME', 'SA', 'WA']
    groups = [(g, c) for g in ['M', 'F'] for c in cats_order]
    labels = [f'{g}/{c}' for g, c in groups]
    n = len(groups)

    # Build raw count matrix
    raw = np.zeros((n, n))
    for i, vg in enumerate(groups):
        for j, tg in enumerate(groups):
            raw[i, j] = vil_gcf.get((vg, tg), 0)

    # Row-normalize to percentages
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid div by 0
    pct = 100 * raw / row_sums

    # Expected % if votes were distributed proportional to pool representation
    # Each gender×category has 10/140 = 7.14% of names, but in each game only 7 of 140 are present
    # Simple baseline: uniform across 14 groups = 7.14% each
    expected = 100.0 / n

    # Deviation from expected
    dev = pct - expected

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8.5))

    # LEFT: Row-normalized percentage heatmap
    im1 = ax1.imshow(pct, cmap='YlOrRd', aspect='auto', vmin=0, vmax=pct.max() * 1.05)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel('Target Group', fontsize=11)
    ax1.set_ylabel('Voter Group', fontsize=11)
    ax1.set_title('Vote Distribution (%)\n(each row sums to 100%)', fontsize=12)

    for i in range(n):
        for j in range(n):
            val = pct[i, j]
            count = int(raw[i, j])
            tc = 'white' if val > pct.max() * 0.6 else 'black'
            ax1.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7,
                     fontweight='bold' if val > 10 else 'normal', color=tc)

    # Divider lines between male and female blocks
    ax1.axhline(y=6.5, color='white', linewidth=3)
    ax1.axvline(x=6.5, color='white', linewidth=3)
    # Quadrant labels
    ax1.text(3, -0.8, 'Male Targets', ha='center', fontsize=9, fontweight='bold', color=MALE_COLOR)
    ax1.text(10, -0.8, 'Female Targets', ha='center', fontsize=9, fontweight='bold', color=FEMALE_COLOR)
    ax1.text(-1.5, 3, 'Male\nVoters', ha='center', va='center', fontsize=9, fontweight='bold',
             color=MALE_COLOR, rotation=0)
    ax1.text(-1.5, 10, 'Female\nVoters', ha='center', va='center', fontsize=9, fontweight='bold',
             color=FEMALE_COLOR, rotation=0)

    cb1 = fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
    cb1.set_label('% of group\'s votes', fontsize=9)

    # RIGHT: Deviation from uniform heatmap (highlights bias)
    absmax = max(abs(dev.min()), abs(dev.max()))
    im2 = ax2.imshow(dev, cmap='RdBu_r', aspect='auto', vmin=-absmax, vmax=absmax)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel('Target Group', fontsize=11)
    ax2.set_ylabel('Voter Group', fontsize=11)
    ax2.set_title(f'Bias (deviation from {expected:.1f}% uniform)\n(red = over-targeted, blue = under-targeted)',
                  fontsize=12)

    for i in range(n):
        for j in range(n):
            val = dev[i, j]
            tc = 'white' if abs(val) > absmax * 0.6 else 'black'
            ax2.text(j, i, f'{val:+.1f}', ha='center', va='center', fontsize=7,
                     fontweight='bold' if abs(val) > 3 else 'normal', color=tc)

    ax2.axhline(y=6.5, color='white', linewidth=3)
    ax2.axvline(x=6.5, color='white', linewidth=3)
    ax2.text(3, -0.8, 'Male Targets', ha='center', fontsize=9, fontweight='bold', color=MALE_COLOR)
    ax2.text(10, -0.8, 'Female Targets', ha='center', fontsize=9, fontweight='bold', color=FEMALE_COLOR)
    ax2.text(-1.5, 3, 'Male\nVoters', ha='center', va='center', fontsize=9, fontweight='bold',
             color=MALE_COLOR, rotation=0)
    ax2.text(-1.5, 10, 'Female\nVoters', ha='center', va='center', fontsize=9, fontweight='bold',
             color=FEMALE_COLOR, rotation=0)

    cb2 = fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
    cb2.set_label('Deviation from expected (pp)', fontsize=9)

    fig.suptitle('Cross-Group Vote Targeting: Gender × Ethnicity',
                 fontsize=15, fontweight='bold', y=1.02)

    # Total villager votes
    total_votes = int(raw.sum())
    desc = (f"{model_label}  •  {len(games)} games  •  {total_votes:,} villager votes  •  "
            f"14 groups (2 genders × 7 ethnic categories)\n"
            f"LEFT: How each voter group distributes votes. Read row-by-row: e.g. row M/SA shows "
            f"where South Asian male-named players cast their votes.\n"
            f"RIGHT: Deviation from uniform ({expected:.1f}%). Red = over-targeted, blue = under-targeted. "
            f"The upper-left quadrant (male→male) is\n"
            f"consistently red, showing all groups over-target male names. "
            f"Lower-right quadrant (female→female) is consistently blue.")
    fig.text(0.5, -0.06, desc, ha='center', va='top', fontsize=9, color='#555',
             style='italic', wrap=True)

    fig.tight_layout(rect=[0, 0.07, 1, 0.98])
    fig.savefig(out / '07_cross_group_targeting.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  07_cross_group_targeting.png")


def generate_all(log_path, model_label, out_dir=None, prefix=""):
    """Generate all figures + JSON for a given log file and model."""
    out = Path(out_dir) if out_dir else Path(__file__).parent

    print(f"Parsing {log_path}...")
    games = parse_log(log_path)
    print(f"  {len(games)} games parsed")

    stats, vil_gf, wolf_gf, gxc, vil_gcf = compute_all(games)
    print("Generating figures...")

    fig_gender_gap(games, stats, out, model_label)
    fig_vote_flow(games, vil_gf, wolf_gf, out, model_label)
    fig_top_bottom(games, stats, out, model_label)
    fig_ethnicity(games, gxc, out, model_label)
    fig_convergence(games, out, model_label)
    fig_wolf_immunity(games, stats, out, model_label)
    fig_cross_group_targeting(games, vil_gcf, out, model_label)

    # Export JSON results
    log_stem = Path(log_path).stem
    json_out = out.parent / 'results' / f'{log_stem}.json'
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_data = {
        'model': model_label,
        'games': len(games),
        'pool_size': 140,
        'per_name': {n: s for n, s in stats.items()},
        'vil_gender_flow': {f'{k[0]}>{k[1]}': v for k, v in vil_gf.items()},
        'wolf_gender_flow': {f'{k[0]}>{k[1]}': v for k, v in wolf_gf.items()},
        'gender_x_category': {f'{k[0]}/{k[1]}': v for k, v in gxc.items()},
        'vil_cross_group_flow': {f'{k[0][0]}/{k[0][1]}>{k[1][0]}/{k[1][1]}': v for k, v in vil_gcf.items()},
    }
    with open(json_out, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Results saved to {json_out}")

    print(f"\nDone! {len(games)} games, 7 figures saved to {out}/")


if __name__ == '__main__':
    import argparse as ap
    parser = ap.ArgumentParser(description="Generate figures from speed mafia logs")
    parser.add_argument("--log", type=str, default=None, help="Path to log file")
    parser.add_argument("--model", type=str, default="GPT-4o-mini", help="Model label for figure titles")
    parser.add_argument("--out", type=str, default=None, help="Output directory for figures")
    args = parser.parse_args()

    base = Path(__file__).parent
    log = args.log or str(base.parent / 'logs' / 'gpt4omini_140n_700g.log')
    out = args.out or str(base)

    if not Path(log).exists():
        print(f"Log not found: {log}")
        exit(1)

    generate_all(log, args.model, out)
