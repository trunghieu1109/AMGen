async def forward_160(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Clarify physical meaning and factor via Debate
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_1 = (
        "Sub-task 1: Clarify the physical meaning of λ₁ (gas–gas mean free path) and λ₂ (electron–gas mean free path), "
        "confirm that direct comparison is valid, and interpret the significance of the factor 1.22." + debate_instr
    )
    debate_agents_1 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                       for role in self.debate_role]
    N_max_1 = self.max_round
    all_thinking1 = [[] for _ in range(N_max_1)]
    all_answer1 = [[] for _ in range(N_max_1)]
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":debate_instruction_1,"context":["user query"],"agent_collaboration":"Debate"}
    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking1, answer1 = await agent([taskInfo], debate_instruction_1, r, is_sub_task=True)
            else:
                thinking1, answer1 = await agent([taskInfo] + all_thinking1[r-1] + all_answer1[r-1], debate_instruction_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking1[r].append(thinking1)
            all_answer1[r].append(answer1)
    final_instr_1 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1(
        [taskInfo] + all_thinking1[-1] + all_answer1[-1],
        "Sub-task 1: Clarify meaning and significance." + final_instr_1,
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2, Sub-task 2: Derive λ₁ via SC_CoT
    cot_sc_instruction_2 = (
        "Sub-task 2: Derive the expression for λ₁ = 1/(√2·n·σ₁) for molecule–molecule collisions, explicitly including the √2 prefactor."
    )
    N2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                    for _ in range(N2)]
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction_2,
                     "context":["user query","thinking1","answer1"],"agent_collaboration":"SC_CoT"}
    possible_thinks2 = []
    possible_ans2 = []
    for i in range(N2):
        thinking2_i, answer2_i = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinks2.append(thinking2_i)
        possible_ans2.append(answer2_i)
    final_decision_agent_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2(
        [taskInfo, thinking1, answer1] + possible_thinks2 + possible_ans2,
        "Given all the above, synthesize and choose the correct expression for λ₁.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 2, Sub-task 3: Derive λ₂ via SC_CoT
    cot_sc_instruction_3 = (
        "Sub-task 3: Derive the expression for λ₂ for electron–gas collisions, incorporating the velocity averaging factor C≈0.86–0.90."
    )
    cot_agents_3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                    for _ in range(N2)]
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_sc_instruction_3,
                     "context":["user query","thinking1","answer1"],"agent_collaboration":"SC_CoT"}
    possible_thinks3 = []
    possible_ans3 = []
    for i in range(N2):
        thinking3_i, answer3_i = await cot_agents_3[i]([taskInfo, thinking1, answer1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_thinks3.append(thinking3_i)
        possible_ans3.append(answer3_i)
    final_decision_agent_3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3(
        [taskInfo, thinking1, answer1] + possible_thinks3 + possible_ans3,
        "Given all the above, synthesize and choose the correct expression for λ₂.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 3, Sub-task 4: Gather cross sections via Reflexion
    reflect_inst = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction_4 = (
        "Sub-task 4: Gather and justify typical values for σ₁ (~10⁻¹⁹ m²) and σ₂ (~10⁻¹⁹ to 10⁻²⁰ m²) at 1000 kV from tabulated sources." + reflect_inst
    )
    cot_agent_4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_reflect_instruction_4,
                     "context":["user query","thinking2","answer2","thinking3","answer3"],"agent_collaboration":"Reflexion"}
    cot_inputs_4 = [taskInfo, thinking2, answer2, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    critic_inst_4 = (
        "Please review the answer above and criticize on where might be wrong. "
        "If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
    )
    for i in range(self.max_round):
        feedback4, correct4 = await critic_agent_4([taskInfo, thinking4, answer4], critic_inst_4, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, feedback: {feedback4.content}; correct: {correct4.content}")
        if correct4.content == "True":
            break
        cot_inputs_4 += [thinking4, answer4, feedback4]
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 3, Sub-task 5: Compute ratio via Debate
    debate_instr_5 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_5 = (
        "Sub-task 5: Compute the ratio R = λ₂/λ₁ = (C·σ₁)/(√2·σ₂), estimate its numeric range, and verify that 1 < R < 1.22." + debate_instr_5
    )
    debate_agents_5 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                       for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":debate_instruction_5,
                     "context":["user query","thinking4","answer4"],"agent_collaboration":"Debate"}
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                thinking5, answer5 = await agent(
                    [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1],
                    debate_instruction_5, r, is_sub_task=True
                )
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_instr_5 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_5 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5(
        [taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1],
        "Sub-task 5: Compute R and verify range." + final_instr_5,
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    # Stage 4, Sub-task 6: Select final choice via Reflexion
    reflect_inst_6 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction_6 = (
        "Sub-task 6: Compare the computed ratio R against the four answer choices, select the option corresponding to 1 < λ₂/λ₁ < 1.22, and justify the inequality." + reflect_inst_6
    )
    cot_agent_6 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {"subtask_id":"subtask_6","instruction":cot_reflect_instruction_6,
                     "context":["user query","thinking5","answer5"],"agent_collaboration":"Reflexion"}
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, thinking: {thinking6.content}; answer: {answer6.content}")
    critic_inst_6 = (
        "Please review the answer above and criticize on where might be wrong. "
        "If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
    )
    for i in range(self.max_round):
        feedback6, correct6 = await critic_agent_6([taskInfo, thinking6, answer6], critic_inst_6, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, feedback: {feedback6.content}; correct: {correct6.content}")
        if correct6.content == "True":
            break
        cot_inputs_6 += [thinking6, answer6, feedback6]
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs