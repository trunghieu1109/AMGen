async def forward_190(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the starting material and each reagent's role and expected chemical transformation in the sequence. "
        "Explicitly identify the functional groups present and the likely site of reaction for each reagent. Avoid assumptions about downstream steps; focus on accurate reagent-function mapping to prevent mechanistic errors later. "
        "Use the provided query for context."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Summarize the structural changes expected after each reaction step, focusing on functional group transformations and substituent modifications. "
        "Track the fate of key substituents (e.g., prop-1-en-2-yl) and newly introduced groups (e.g., benzyl ether). "
        "This summary will serve as a foundation for mechanistic analysis and must avoid oversimplifications. "
        "Use outputs from Sub-task 1 as input and context."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_reflect_instruction3 = (
        "Sub-task 3: Perform a detailed mechanistic analysis of the n-butyllithium treatment of the tosyl hydrazone intermediate (product 2). "
        "Explicitly write out the stepwise Shapiro reaction mechanism, including deprotonation, formation of vinyl lithium intermediate, elimination of tosyl group and nitrogen gas, and protonation steps. "
        "Identify all leaving groups and confirm that no butyl group is transferred to the ring. Provide intermediate structures or detailed descriptions. "
        "Filter the valid scenarios and rigorously cross-validate. Use outputs from Sub-tasks 1 and 2 as input and context."
    )
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_instruction4 = (
        "Sub-task 4: Integrate the mechanistic insights from the Shapiro reaction step with the subsequent catalytic hydrogenation step to predict the structure of product 4. "
        "Analyze how the exocyclic or endocyclic alkene formed after n-BuLi treatment is saturated, and how the prop-1-en-2-yl substituent is converted to isopropyl. "
        "Avoid assumptions of new substituent installation (e.g., butyl group). Use all prior structural summaries and mechanistic details to support the prediction. "
        "Use outputs from Sub-task 3 and Sub-task 2 as input and context."
    )
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 2", "answer of subtask 2"]
    }
    results4, log4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Select the correct product structure from the given choices based on the predicted structure of product 4. "
        "Provide a detailed rationale referencing the mechanistic steps, intermediate structures, and transformations. "
        "Explicitly explain why other choices are inconsistent with the reaction sequence and mechanistic understanding, especially addressing the incorrect assumption of butyl substitution in choice 4. "
        "Use outputs from Sub-task 4 as input and context."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
