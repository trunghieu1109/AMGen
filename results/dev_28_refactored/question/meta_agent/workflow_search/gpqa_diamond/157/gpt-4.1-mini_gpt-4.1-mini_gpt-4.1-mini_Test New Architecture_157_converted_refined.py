async def forward_157(self, taskInfo):
    logs = []

    cot_instruction_1_1 = (
        "Sub-task 1: Analyze and characterize the functional roles of the transcription factor domains (transactivation and dimerization), "
        "the phosphorylation activation mechanism, and the effects of mutations X and Y. Specifically, clarify the molecular mechanism by which mutation Y in the dimerization domain acts as a dominant-negative mutation, "
        "emphasizing that dominant-negative mutations typically retain binding to wild-type proteins and interfere with their function rather than abolishing dimerization."
    )
    cot_agent_desc_1_1 = {
        'instruction': cot_instruction_1_1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ['user query']
    }
    results_1_1, log_1_1 = await self.sc_cot(
        subtask_id='stage_1.subtask_1',
        cot_agent_desc=cot_agent_desc_1_1,
        n_repeat=self.max_sc
    )
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Identify and explicitly define the mechanistic requirements and typical molecular consequences of dominant-negative mutations in protein dimerization domains, "
        "including the necessity of mutant-wild-type binding, potential for nonfunctional complex formation or aggregation, and resulting functional outcomes. "
        "This subtask must incorporate feedback to avoid assumptions that dominant-negative mutations cause loss of dimerization and to highlight common dominant-negative mechanisms such as sequestration or poisoning of wild-type function."
    )
    final_decision_instruction_1_2 = (
        "Sub-task 2: Synthesize and choose the most consistent mechanistic framework for dominant-negative mutations in dimerization domains."
    )
    cot_agent_desc_1_2 = {
        'instruction': cot_instruction_1_2,
        'final_decision_instruction': final_decision_instruction_1_2,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_1_2, log_1_2 = await self.sc_cot(
        subtask_id='stage_1.subtask_2',
        cot_agent_desc=cot_agent_desc_1_2,
        n_repeat=self.max_sc
    )
    logs.append(log_1_2)

    cot_instruction_2_1 = (
        "Sub-task 1: Map each of the four provided molecular phenotype answer choices onto the mechanistic framework established in stage_1.subtask_2. "
        "For each choice, assess whether it meets the mechanistic criteria of a dominant-negative mutation (e.g., retention of binding, interference with wild-type function) and identify any internal contradictions or ambiguous phrasing (such as 'loss of dimerization' combined with 'wild-type phenotype')."
    )
    cot_agent_desc_2_1 = {
        'instruction': cot_instruction_2_1,
        'input': [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results_2_1, log_2_1 = await self.sc_cot(
        subtask_id='stage_2.subtask_1',
        cot_agent_desc=cot_agent_desc_2_1,
        n_repeat=self.max_sc
    )
    logs.append(log_2_1)

    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Conduct a critical self-reflection and validation checkpoint on the mapping results from subtask_1. "
        "Explicitly flag any answer choices containing contradictory or ambiguous language that conflicts with the mechanistic understanding of dominant-negative mutations. "
        "If no answer perfectly fits, identify the closest option and clearly document the reasoning and discrepancies. "
        "This step addresses previous failures to recognize contradictions and prevents premature acceptance of misleading options."
    )
    critic_instruction_2_2 = (
        "Please review and provide the limitations of provided solutions of stage_2.subtask_1, focusing on contradictions or ambiguities in answer choices related to dominant-negative mutation mechanisms."
    )
    cot_reflect_desc_2_2 = {
        'instruction': cot_reflect_instruction_2_2,
        'critic_instruction': critic_instruction_2_2,
        'input': [taskInfo, results_1_2['thinking'], results_1_2['answer'], results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_2_2, log_2_2 = await self.reflexion(
        subtask_id='stage_2.subtask_2',
        reflect_desc=cot_reflect_desc_2_2,
        n_repeat=self.max_round
    )
    logs.append(log_2_2)

    debate_instruction_3_1 = (
        "Sub-task 1: Engage in a structured Debate among agents to synthesize the mechanistic analysis, phenotype mapping, and validation results. "
        "Select the most likely molecular phenotype observed in the presence of mutation Y, ensuring the choice aligns fully with the dominant-negative mechanism and contains no internal contradictions. "
        "If no option is perfect, select the best fit while transparently noting any limitations or ambiguities."
    )
    final_decision_instruction_3_1 = (
        "Sub-task 1: Select the most likely molecular phenotype observed in the presence of mutation Y based on all prior analyses."
    )
    debate_desc_3_1 = {
        'instruction': debate_instruction_3_1,
        'final_decision_instruction': final_decision_instruction_3_1,
        'input': [taskInfo, results_2_2['thinking'], results_2_2['answer']],
        'context_desc': ['user query', 'thinking of stage_2.subtask_2', 'answer of stage_2.subtask_2'],
        'temperature': 0.5
    }
    results_3_1, log_3_1 = await self.debate(
        subtask_id='stage_3.subtask_1',
        debate_desc=debate_desc_3_1,
        n_repeat=self.max_round
    )
    logs.append(log_3_1)

    aggregate_instruction_3_2 = (
        "Sub-task 2: Aggregate and finalize the reasoning and answer selection, producing a clear, logically consistent explanation that integrates all prior analyses and explicitly addresses the feedback from previous failures. "
        "Confirm that the final answer choice is justified by the mechanistic understanding and that any discrepancies in answer options are acknowledged."
    )
    aggregate_desc_3_2 = {
        'instruction': aggregate_instruction_3_2,
        'input': [taskInfo, results_3_1['thinking'], results_3_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_3.subtask_1', 'answer of stage_3.subtask_1']
    }
    results_3_2, log_3_2 = await self.aggregate(
        subtask_id='stage_3.subtask_2',
        aggregate_desc=aggregate_desc_3_2
    )
    logs.append(log_3_2)

    final_answer = await self.make_final_answer(results_3_2['thinking'], results_3_2['answer'])
    return final_answer, logs
