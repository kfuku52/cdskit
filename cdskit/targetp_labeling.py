def strict_uniprot_targetp_label(location_text, organism_group):
    txt = str(location_text or '').lower()
    organism_group = str(organism_group or '').strip().lower()
    if txt.strip() == '':
        return None, 'missing_location_text'

    has_sp = ('secreted' in txt) or ('signal peptide' in txt)
    has_mtp = 'mitochond' in txt
    has_ctp = ('chloroplast' in txt) or ('plastid' in txt)
    has_thylakoid = 'thylakoid' in txt
    has_lumen = ('lumen' in txt) or ('lumenal' in txt) or ('luminal' in txt)

    if organism_group != 'plant' and (has_ctp or has_thylakoid):
        return None, 'nonplant_plastid'

    plastid_signal = organism_group == 'plant' and has_ctp
    ltp_signal = organism_group == 'plant' and has_thylakoid and has_lumen
    if sum(1 for value in [has_sp, has_mtp, plastid_signal] if value) > 1:
        return None, 'ambiguous'
    if ltp_signal and (has_sp or has_mtp):
        return None, 'ambiguous'

    if ltp_signal:
        if 'membrane' in txt and 'thylakoid lumen' not in txt:
            return None, 'ltp_membrane_noise'
        return 'lTP', ''
    if plastid_signal:
        if has_thylakoid:
            return None, 'thylakoid_not_lumen'
        return 'cTP', ''
    if has_mtp:
        return 'mTP', ''
    if has_sp:
        return 'SP', ''
    return 'noTP', ''
