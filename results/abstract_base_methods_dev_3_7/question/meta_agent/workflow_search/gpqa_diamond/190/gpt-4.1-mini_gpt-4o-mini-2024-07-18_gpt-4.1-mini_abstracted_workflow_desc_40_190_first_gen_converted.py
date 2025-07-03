async def forward_190(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Subtask 1: Analyze the starting material 3-(hydroxymethyl)-5-(prop-1-en-2-yl)cyclohexan-1-one and determine the structure of product 1 after treatment with sodium hydride and benzyl bromide (alkylation of the hydroxyl group)."
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_instruction=cot_instruction_1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context="user input"
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing starting material and alkylation, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction_2 = "Subtask 2: Based on product 1, analyze the reaction with p-toluenesulfonyl hydrazide and catalytic HCl to form product 2 (formation of tosylhydrazone derivative)."
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_instruction=cot_sc_instruction_2,
        input_list=[taskInfo, results1['thinking'], results1['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 1", "answer of subtask 1"],
        n_repeat=self.max_sc
    )
    for idx, _ in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, analyzing tosylhydrazone formation, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_sc_instruction_3 = "Subtask 3: Based on product 2, analyze the treatment with n-butyllithium at low temperature followed by aqueous ammonium chloride to form product 3 (Shapiro reaction leading to alkene formation or related transformation)."
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_instruction=cot_sc_instruction_3,
        input_list=[taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        n_repeat=self.max_sc
    )
    for idx, _ in enumerate(results3['list_thinking']):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, analyzing Shapiro reaction, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    debate_instruction_4 = "Subtask 4: Analyze the catalytic hydrogenation of product 3 with Pd/C under hydrogen atmosphere to form product 4 (reduction of double bonds or other reducible groups)."
    final_decision_instruction_4 = "Subtask 4: Make final decision on the structure of product 4 after hydrogenation."
    debate_desc_4 = {
        "instruction": debate_instruction_4,
        "context": ["user query", results3['thinking'], results3['answer']],
        "input": [taskInfo, results3['thinking'], results3['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_4 = {
        "instruction": final_decision_instruction_4,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc_4,
        final_decision_desc=final_decision_desc_4,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, analyzing hydrogenation, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding product 4 structure, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    debate_instruction_5 = "Subtask 5: Compare the deduced structure of product 4 with the given multiple-choice options and select the correct answer."
    final_decision_instruction_5 = "Subtask 5: Make final decision on the correct multiple-choice answer for product 4's structure."
    debate_desc_5 = {
        "instruction": debate_instruction_5,
        "context": ["user query", results4['thinking'], results4['answer']],
        "input": [taskInfo, results4['thinking'], results4['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_5 = {
        "instruction": final_decision_instruction_5,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc_5,
        final_decision_desc=final_decision_desc_5,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results5['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, comparing product 4 with choices, thinking: {results5['list_thinking'][round][idx].content}; answer: {results5['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, selecting correct choice, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
