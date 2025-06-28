async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    """
    [Stage 0: Compute Energy Uncertainties]
    [Objective] 
    - Calculate the energy uncertainty (ΔE) for each quantum state using their given lifetimes (10^-9 sec and 10^-8 sec) based on the energy-time uncertainty principle ΔE ≈ ħ / Δt.
    - This provides a quantitative measure of the minimum energy spread for each state.
    [Agent Collaborations]
    - Use Chain-of-Thought (CoT) to explicitly calculate and reason about the energy uncertainties.
    """
    cot_instruction = (
        "Sub-task 1: Calculate the energy uncertainty (ΔE) for each quantum state with lifetimes 10^-9 sec and 10^-8 sec "
        "using the energy-time uncertainty principle ΔE ≈ ħ / Δt. Provide detailed calculations and results."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": [taskInfo]
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, calculating energy uncertainties, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    """
    [Stage 1: Analyze Minimum Energy Difference Required]
    [Objective] 
    - Analyze the calculated energy uncertainties from subtask_1 to determine the minimum energy difference required to clearly distinguish the two energy levels.
    - Understand that the energy difference must be larger than the larger of the two uncertainties to resolve the levels.
    [Agent Collaborations]
    - Use Self-Consistency Chain-of-Thought (SC-CoT) to consider multiple reasoning paths and ensure robust analysis.
    """
    cot_sc_instruction = (
        "Sub-task 2: Based on the energy uncertainties calculated in Sub-task 1, analyze and determine the minimum energy difference "
        "required to clearly distinguish the two quantum states. Consider that the energy difference must exceed the larger uncertainty."
    )
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction,
        "context": [taskInfo, thinking1, answer1]
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, analyzing minimum energy difference, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Compare and Select Suitable Energy Difference Option]
    [Objective] 
    - Compare the given multiple-choice energy difference options (10^-9 eV, 10^-11 eV, 10^-8 eV, 10^-4 eV) against the minimum energy difference required from subtask_2.
    - Identify which options satisfy the condition for clear resolution of the two energy levels.
    - Select the best option that meets the criterion.
    [Agent Collaborations]
    - Use Reflexion to iteratively evaluate and select the best energy difference option based on the comparison.
    """
    cot_reflect_instruction = (
        "Sub-task 3: Compare the given energy difference options (10^-9 eV, 10^-11 eV, 10^-8 eV, 10^-4 eV) against the minimum energy difference "
        "required to resolve the two quantum states clearly, and select the best option that satisfies this condition."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction,
        "context": cot_inputs
    }
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, comparing and selecting energy difference option, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3],
                                             "Review the selection of energy difference option and provide feedback on its validity and completeness.",
                                             i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining selection, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
