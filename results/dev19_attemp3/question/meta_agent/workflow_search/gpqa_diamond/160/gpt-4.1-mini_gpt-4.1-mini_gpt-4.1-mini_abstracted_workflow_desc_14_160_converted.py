async def forward_160(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_1 = "Sub-task 1: Explicitly extract and clarify the physical definitions and measurement contexts of λ1 and λ2. Emphasize that λ1 is the classical mean free path of gas molecules determined by gas-gas collisions under ultra-high vacuum conditions, while λ2 relates to the mean free path associated with electron scattering off gas molecules. Identify whether λ2 refers to the electron mean free path or an effective mean free path of gas molecules influenced by the electron beam. Address the ambiguity in the problem statement and avoid conflating these distinct quantities. This subtask must prevent the previous error of treating λ1 and λ2 as directly comparable without distinction. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_1 = []
    all_answer_1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_1):
        thinking1, answer1 = await agent([taskInfo], debate_instr_1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, clarifying definitions of λ1 and λ2, thinking: {thinking1.content}; answer: {answer1.content}")
        all_thinking_1.append(thinking1)
        all_answer_1.append(answer1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + all_thinking_1 + all_answer_1, "Sub-task 1: Synthesize clarified definitions and measurement contexts of λ1 and λ2. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing definitions of λ1 and λ2, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Based on the output from Sub-task 1, analyze and derive or reference the relevant physical parameters governing λ1 and λ2, specifically the scattering cross-sections for gas-gas collisions and electron-gas molecule collisions at 1000 kV accelerating voltage. Calculate or cite realistic values or formulae for these cross-sections and their ratios. Avoid unsupported assumptions such as the unexplained factor 1.22. This subtask must ground the relationship between λ1 and λ2 in actual physics rather than guesswork, addressing the previous failure to justify quantitative factors."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing scattering cross-sections and their ratios, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings_2 + possible_answers_2, "Sub-task 2: Synthesize and choose the most consistent and physically grounded relationship between λ1 and λ2 based on scattering cross-sections.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing scattering cross-section analysis, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_3 = "Sub-task 3: Evaluate the physical implications of electron-gas molecule scattering on the effective mean free path, using the clarified definitions and derived cross-sections from previous subtasks. Compare λ2 to λ1 based on their distinct physical meanings and the quantitative relationship of their scattering cross-sections. Interpret whether λ2 should be greater than, less than, or equal to λ1, and explain the reasoning without conflating the two mean free paths. This subtask must avoid the previous error of oversimplified assumptions and must critically assess the impact of electron beam presence on mean free path measurements. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_3 = []
    all_answer_3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instr_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_3):
        thinking3, answer3 = await agent([taskInfo, thinking1, answer1, thinking2, answer2], debate_instr_3, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, evaluating physical implications on mean free path, thinking: {thinking3.content}; answer: {answer3.content}")
        all_thinking_3.append(thinking3)
        all_answer_3.append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking_3 + all_answer_3, "Sub-task 3: Synthesize evaluation of electron beam impact on mean free path and compare λ2 and λ1. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing physical implications on mean free path, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    debate_instr_4 = "Sub-task 4: Select the correct conclusion about λ2 from the given multiple-choice options based on the rigorous analysis and reasoning from prior subtasks. Ensure that the choice is justified by the physical distinctions and quantitative relationships established earlier, explicitly rejecting unsupported options such as those relying on the unexplained factor 1.22. This subtask should synthesize all prior findings and provide a clear, well-supported final answer. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_4 = []
    all_answer_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instr_4,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_4):
        thinking4, answer4 = await agent([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], debate_instr_4, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, selecting correct conclusion about λ2, thinking: {thinking4.content}; answer: {answer4.content}")
        all_thinking_4.append(thinking4)
        all_answer_4.append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3] + all_thinking_4 + all_answer_4, "Sub-task 4: Provide the final justified answer selecting the correct conclusion about λ2 from the given options. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing answer about λ2, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
