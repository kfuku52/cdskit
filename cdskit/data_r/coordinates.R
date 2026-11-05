map_trimmed_to_untrimmed_nongap_index = function(seq_trim, seq_untrim, gap_code='04') {
  seq_trim = as.character(seq_trim)
  seq_untrim = as.character(seq_untrim)
  out = rep(NA_integer_, length(seq_trim))
  if (length(seq_trim)==0 || length(seq_untrim)==0) {
    return(out)
  }
  trim_non_gap_idx = which(seq_trim != gap_code)
  untrim_non_gap_idx = which(seq_untrim != gap_code)
  if (length(trim_non_gap_idx)==0 || length(untrim_non_gap_idx)==0) {
    return(out)
  }
  trim_tokens = seq_trim[trim_non_gap_idx]
  untrim_tokens = seq_untrim[untrim_non_gap_idx]
  u_ptr = 1L
  for (i in seq_along(trim_tokens)) {
    tok = trim_tokens[i]
    while (u_ptr <= length(untrim_tokens) && untrim_tokens[u_ptr] != tok) {
      u_ptr = u_ptr + 1L
    }
    if (u_ptr > length(untrim_tokens)) {
      return(rep(NA_integer_, length(seq_trim)))
    }
    out[trim_non_gap_idx[i]] = u_ptr
    u_ptr = u_ptr + 1L
  }
  return(out)
}

intron_site_data = function(tips, seqs) {
    seqs = seqs[names(seqs) %in% as.character(tips[['label']])]
    if (!length(seqs)) return(NULL)
    if (any(nchar(seqs) == 0) || length(unique(nchar(seqs))) != 1) stop('Intron sites require an equal-length untrimmed CDS alignment')
    # The accepted alphabet is ASCII; byte matching keeps the same validation
    # while avoiding locale-aware regex work on every alignment character.
    if (any(grepl('[^ACGTRYSWKMBDHVN?.-]', seqs,perl=TRUE,useBytes=TRUE))) stop('Intron sites require nucleotide sequences')
    events = list()
    rows = list()
    diagnostics = list()
    for (i in seq_len(nrow(tips))) {
        id = as.character(tips[['label']][i])
        reason = NA_character_
        if (!(id %in% names(seqs))) reason = 'missing_alignment'
        count = tips[['num_intron']][i]
        length_cds = tips[['intron_feature_size']][i]
        phase = tips[['cds_first_phase']][i]
        trans = 'splice_mode' %in% names(tips) &&
            !is.na(tips[['splice_mode']][i]) && tips[['splice_mode']][i] == 'trans-splicing'
        if ((!trans && is.na(count)) || is.na(length_cds)) reason = 'missing_GFF'
        if (is.na(reason)) {
            chars = strsplit(seqs[[id]], '', fixed=TRUE)[[1]]
            is_nongap = !chars %in% c('-', '.')
            nongap = which(is_nongap)
            if (length(nongap) != length_cds) {
                # CDSKit may complete the terminal codon with one or two N
                # bases. GFF coordinates and intron offsets describe only the
                # original CDS, so exclude exactly that verified suffix.
                padding = length(nongap) - length_cds
                status = if ('structure_status' %in% names(tips))
                    as.character(tips[['structure_status']][i]) else NA_character_
                if (is.finite(padding) && padding %in% 1:2 &&
                    !is.na(status) && status %in% c('length_compatible', 'sequence_verified') &&
                    all(chars[utils::tail(nongap, padding)] == 'N')) {
                    nongap = nongap[seq_len(length_cds)]
                } else {
                    reason = 'CDS_length_mismatch'
                }
            }
        }
        if (is.na(reason) && trans) reason = 'trans_splicing'
        if ('structure_status' %in% names(tips) && !is.na(tips[['structure_status']][i]) &&
            tips[['structure_status']][i] %in% c('excluded_cds', 'sequence_not_coordinate_matched', 'cds_length_mismatch')) {
            reason = as.character(tips[['structure_status']][i])
        }
        if ('phase_status' %in% names(tips) && !is.na(tips[['phase_status']][i]) &&
            tips[['phase_status']][i] %in% c('conflicting', 'ribosomal-slippage', 'pseudogene', 'source-overlap', 'ordered-fragments')) {
            reason = as.character(tips[['phase_status']][i])
        }
        if ('splice_mode' %in% names(tips) && !is.na(tips[['splice_mode']][i]) &&
            tips[['splice_mode']][i] %in% c('ribosomal-slippage', 'pseudogene', 'source-overlap', 'ordered-fragments')) reason = as.character(tips[['splice_mode']][i])
        if (!is.na(reason)) {
            diagnostics[[length(diagnostics)+1]] = data.frame(node_name=id, reason=reason)
            next
        }
        if (!is.finite(count) || count < 0 || count != floor(count) || (!is.na(phase) && !phase %in% 0:2)) stop('Invalid intron count or CDS phase')
        value = as.character(tips[['intron_positions']][i])
        offsets = if (is.na(value) || !nzchar(value)) numeric() else suppressWarnings(as.numeric(strsplit(value, '[;,]')[[1]]))
        if (length(offsets) != count || any(!is.finite(offsets)) || any(offsets != floor(offsets)) ||
            any(offsets < 1 | offsets >= length_cds) || anyDuplicated(offsets) || is.unsorted(offsets)) stop('Invalid intron offsets for ', id)
        rows[[id]] = list(chars=chars, cumulative=cumsum(is_nongap), phase=phase, y=tips[['y']][i],
            colour=if ('tiplab_color' %in% names(tips)) as.character(tips[['tiplab_color']][i]) else 'black')
        if (length(offsets)) {
            left = nongap[offsets]; right = nongap[offsets+1]
            status = rep(if (is.na(phase)) 'position_only' else 'mapped',length(offsets))
            status[!(chars[left] %in% c('A','C','G','T') & chars[right] %in% c('A','C','G','T'))] = 'ambiguous_bases'
            status[right != left+1] = 'alignment_gap'
            events[[length(events)+1]] = data.frame(node_name=id, intron_index=seq_along(offsets), cds_offset=offsets,
                alignment_left=left, alignment_right=right, phase=(offsets-phase) %% 3, status=status)
        }
    }
    events = if (length(events)) do.call(rbind, events) else data.frame(node_name=character(), intron_index=integer(),
        cds_offset=numeric(), alignment_left=integer(), alignment_right=integer(), phase=numeric(), status=character())
    keys = unique(events[events$status == 'mapped',c('alignment_left','phase'),drop=FALSE])
    unknown = unique(events[events$status == 'position_only',c('alignment_left','phase'),drop=FALSE])
    keys = rbind(keys, unknown[!unknown$alignment_left %in% keys$alignment_left,,drop=FALSE])
    keys = keys[order(keys$alignment_left,keys$phase),,drop=FALSE]
    keys$site_id = sprintf('I%03d',seq_len(nrow(keys)))
    key = function(left,phase) paste(left,phase,sep=':')
    events$site_id = keys$site_id[match(key(events$alignment_left,events$phase),key(keys$alignment_left,keys$phase))]
    events$site_id[events$status != 'mapped'] = NA_character_
    events$candidate_site_ids = rep(NA_character_, nrow(events))
    for (k in which(events$status == 'position_only')) {
        candidates = keys$site_id[keys$alignment_left == events$alignment_left[k]]
        events$candidate_site_ids[k] = paste(candidates,collapse=';')
        if (length(candidates) == 1) events$site_id[k] = candidates
    }
    cells = data.frame()
    if (nrow(keys)) {
        mapped = events$status == 'mapped' & !is.na(events$site_id)
        mapped_by_gene = split(events$site_id[mapped], events$node_name[mapped])
        possible = events$status == 'position_only'
        possible_by_gene = split(events$alignment_left[possible], events$node_name[possible])
        b = keys$alignment_left
        row_ids = names(rows)
        site_count = nrow(keys)
        states = character(length(rows)*site_count)
        x = seq_len(site_count)
        for (i in seq_along(rows)) {
            id = row_ids[i]
            row = rows[[id]]
            available = row$chars[b] %in% c('A','C','G','T') &
                row$chars[b+1] %in% c('A','C','G','T') &
                (is.na(row$phase) | is.na(keys$phase) | (row$cumulative[b]-row$phase) %% 3 == keys$phase)
            state = rep('unresolved', site_count)
            state[available] = 'absent'
            # An unknown phase still marks every candidate at this boundary;
            # an exact mapped event takes precedence over availability.
            state[b %in% possible_by_gene[[id]]] = 'position_only'
            state[keys$site_id %in% mapped_by_gene[[id]]] = 'present'
            states[(i-1)*site_count+x] = state
        }
        cells = data.frame(node_name=rep(row_ids,each=site_count),
            y=rep(unlist(lapply(rows,'[[','y'),use.names=FALSE),each=site_count),
            x=rep(x,length(rows)),site_id=rep(keys$site_id,length(rows)),state=states,
            colour=rep(unlist(lapply(rows,'[[','colour'),use.names=FALSE),each=site_count))
    }
    list(sites=keys, events=events, cells=cells,
        diagnostics=if (length(diagnostics)) do.call(rbind,diagnostics) else data.frame(node_name=character(),reason=character()))
}
