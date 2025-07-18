async def forward_157(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all explicit molecular and genetic information from the query, "
        "including the transcription factor activation mechanism, mutation X and Y characteristics, and their domain locations. "
        "Resolve any conflicting interpretations or uncertainties about the sufficiency of information by collaborative debate to ensure a consistent and confident foundational summary. "
        "This avoids the previous error of inconsistent assumptions that undermined downstream reasoning."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Classify the mutations X and Y by type (recessive loss-of-function vs dominant-negative) and analyze their typical molecular consequences separately. "
        "Explicitly distinguish the recessive loss-of-function mutation X from the dominant-negative mutation Y to avoid ambiguity. "
        "This subtask addresses prior confusion by clarifying mutation types and their expected effects before deeper mechanistic analysis."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=cot_agent_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Enumerate and compare all plausible molecular mechanisms by which a dominant-negative mutation in the dimerization domain (mutation Y) can affect protein function. "
        "Consider mechanisms such as: (a) formation of nonfunctional heterodimers (classic poisoning), (b) misfolding and aggregation sequestering wild-type protein, and (c) accelerated degradation of mutant and/or wild-type proteins. "
        "Map each mechanism to the molecular phenotypes described in the answer choices. "
        "This explicit enumeration and mapping prevents premature assumptions and ensures comprehensive mechanistic coverage."
    )
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=cot_agent_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_instruction4 = (
        "Sub-task 4: Critically evaluate each of the four provided molecular phenotype options against the mechanistic insights established in subtask_3. "
        "Explicitly eliminate each option by matching its molecular signature to the dominant-negative mechanisms supported by the mutation’s description and domain location. "
        "Avoid assuming classic models without justification. "
        "This step ensures rigorous, evidence-based selection of the most likely molecular phenotype observed with mutation Y, addressing prior failures to reconcile mutation location with phenotype."
    )
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=cot_agent_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
