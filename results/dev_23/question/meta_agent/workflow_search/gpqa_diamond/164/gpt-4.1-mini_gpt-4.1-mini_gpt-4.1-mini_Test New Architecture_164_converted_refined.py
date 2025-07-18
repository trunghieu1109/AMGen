async def forward_164(self, taskInfo):
    logs = []

    cot_instruction_1_1 = (
        "Sub-task 1_1: Extract and summarize all essential components and relationships from the query, "
        "explicitly incorporating well-known industrial dual catalyst systems (e.g., Dow’s INSITE, Mitsui’s Borstar) "
        "and chain-walking catalyst mechanisms relevant to ethylene-only polymerization producing branched polyethylene. "
        "This subtask must overcome previous failures by actively applying domain knowledge rather than defaulting to 'insufficient information,' "
        "setting a strong factual foundation for subsequent evaluation."
    )
    cot_agent_desc_1_1 = {
        'instruction': cot_instruction_1_1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_1_1, log_1_1 = await self.debate(
        subtask_id="subtask_1_1",
        debate_desc=cot_agent_desc_1_1,
        n_repeat=self.max_round
    )
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 1_2: Evaluate the correctness of the statement 'Such combined systems are already implemented on an industrial scale in the US.' "
        "Include a two-column elimination table contrasting the statement with specific industrial examples and mechanistic evidence, "
        "ensuring domain knowledge is applied to confirm or refute the claim."
    )
    cot_agent_desc_1_2 = {
        'instruction': cot_instruction_1_2,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1_1", "answer of subtask 1_1"]
    }
    results_1_2, log_1_2 = await self.debate(
        subtask_id="subtask_1_2",
        debate_desc=cot_agent_desc_1_2,
        n_repeat=self.max_round
    )
    logs.append(log_1_2)

    cot_instruction_1_3 = (
        "Sub-task 1_3: Assess the validity of the statement 'Aluminum-based activators do not work for the essential additional reaction step introducing branching.' "
        "Use a two-column elimination table to compare the statement against known activator chemistry and industrial practice, "
        "explicitly addressing previous errors that ignored the effectiveness of alkylaluminum activators in branching catalysis."
    )
    cot_agent_desc_1_3 = {
        'instruction': cot_instruction_1_3,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1_1", "answer of subtask 1_1"]
    }
    results_1_3, log_1_3 = await self.debate(
        subtask_id="subtask_1_3",
        debate_desc=cot_agent_desc_1_3,
        n_repeat=self.max_round
    )
    logs.append(log_1_3)

    cot_instruction_1_4 = (
        "Sub-task 1_4: Analyze the applicability of group VIa transition metal catalysts with specific activators for introducing regular branches in polyethylene. "
        "Include a two-column elimination table contrasting the statement with mechanistic and industrial evidence, "
        "clarifying misconceptions from prior attempts about the typical roles of group VIa catalysts in polyethylene branching."
    )
    cot_agent_desc_1_4 = {
        'instruction': cot_instruction_1_4,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1_1", "answer of subtask 1_1"]
    }
    results_1_4, log_1_4 = await self.debate(
        subtask_id="subtask_1_4",
        debate_desc=cot_agent_desc_1_4,
        n_repeat=self.max_round
    )
    logs.append(log_1_4)

    cot_instruction_1_5 = (
        "Sub-task 1_5: Examine the statement 'Certain noble metal catalysts can be used but are too expensive.' "
        "Produce a two-column elimination table comparing the statement with known catalyst costs and industrial feasibility, "
        "ensuring economic factors are properly integrated into the evaluation."
    )
    cot_agent_desc_1_5 = {
        'instruction': cot_instruction_1_5,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1_1", "answer of subtask 1_1"]
    }
    results_1_5, log_1_5 = await self.debate(
        subtask_id="subtask_1_5",
        debate_desc=cot_agent_desc_1_5,
        n_repeat=self.max_round
    )
    logs.append(log_1_5)

    debate_instruction_2_1 = (
        "Sub-task 2_1: Integrate findings from all evaluation subtasks (1_2 to 1_5) to identify which of the four statements is correct regarding the formation of branched polyethylene using a dual catalyst system and only ethylene. "
        "Synthesize mechanistic, industrial, and economic evidence in a reasoned debate format to confidently select the correct statement, "
        "avoiding previous circular or overly cautious reasoning."
    )
    debate_desc_2_1 = {
        'instruction': debate_instruction_2_1,
        'input': [
            taskInfo,
            results_1_2['thinking'], results_1_2['answer'],
            results_1_3['thinking'], results_1_3['answer'],
            results_1_4['thinking'], results_1_4['answer'],
            results_1_5['thinking'], results_1_5['answer']
        ],
        'temperature': 0.5,
        'context': [
            "user query",
            "thinking of subtask 1_2", "answer of subtask 1_2",
            "thinking of subtask 1_3", "answer of subtask 1_3",
            "thinking of subtask 1_4", "answer of subtask 1_4",
            "thinking of subtask 1_5", "answer of subtask 1_5"
        ],
        'output': ["thinking", "answer"]
    }
    results_2_1, log_2_1 = await self.debate(
        subtask_id="subtask_2_1",
        debate_desc=debate_desc_2_1,
        n_repeat=self.max_round
    )
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1['thinking'], results_2_1['answer'])
    return final_answer, logs
