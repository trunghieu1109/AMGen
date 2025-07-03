async def forward_192(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Subtask 1: Explicitly state the relationship between parallax (plx) and distance (r), including the formula plx = 1/r, "
        "confirm the units of plx and r, and explain the physical meaning of this relationship in the context of the problem."
    )
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_instruction=cot_instruction_1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context="user input"
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing parallax-distance relationship, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_reflect_instruction_2 = (
        "Subtask 2: Express the number of stars per unit distance r by converting from the observed distribution per unit parallax plx, "
        "explicitly including the Jacobian factor: N(r) = N(plx) * |d(plx)/d(r)|. Then substitute plx = 1/r and simplify the expression. "
        "Reflect on the physical meaning of this transformation and ensure the correct interpretation of the distribution."
    )
    cot_reflect_desc_2 = {
        'instruction': cot_reflect_instruction_2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    critic_instruction_2 = (
        "Please review the conversion of the number of stars from per unit parallax to per unit distance, "
        "check for any missing factors such as the Jacobian, and provide feedback on the correctness and limitations."
    )
    critic_desc_2 = {
        'instruction': critic_instruction_2,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results2 = await self.reflexion(
        subtask_id="subtask_2",
        cot_reflect_desc=cot_reflect_desc_2,
        critic_desc=critic_desc_2,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, converting distribution with Jacobian, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results2['critic_agent'].id}, providing feedback, thinking: {results2['list_feedback'][i].content}; answer: {results2['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining conversion, thinking: {results2['list_thinking'][i + 1].content}; answer: {results2['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_sc_instruction_3 = (
        "Subtask 3: Given N(r) = r^5 * (1/r^2), simplify to its final power-law form. "
        "Consider the physical meaning of the distribution and verify the result through multiple reasoning paths."
    )
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_instruction=cot_sc_instruction_3,
        input_list=[taskInfo, results2['thinking'], results2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 2", "answer of subtask 2"],
        n_repeat=self.max_sc
    )
    for idx, _ in enumerate(results3['list_thinking']):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, simplifying power-law dependence, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    debate_instruction_4 = (
        "Subtask 4: Based on the simplified power-law dependence from Subtask 3, debate among multiple agents to select the correct choice among: "
        "A) ~ r^2, B) ~ r^4, C) ~ r^3, D) ~ r^5. Each agent should provide reasoning supporting their choice before voting on the final answer."
    )
    final_decision_instruction_4 = "Subtask 4: Make final decision on the correct power-law dependence of the number of stars on distance r."

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
            agents.append(f"Debate agent {agent.id}, round {round}, debating power-law choice, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")

    agents.append(f"Final Decision agent, selecting final answer, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs
