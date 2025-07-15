async def forward_160(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: Chain-of-Thought
    cot_instruction = (
        "Sub-task 1: Clarify and document the governing physical principles: distinguish between "
        "gas–gas mean free path λ1 (kinetic theory) and electron–gas mean free path λ2 (scattering theory), "
        "explicitly incorporating relativistic corrections at 1000 kV. Avoid assuming cross-section ratios without data."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1:", sub_tasks[-1])

    # Sub-task 2: Self-Consistency Chain-of-Thought
    sc_instruction = (
        "Sub-task 2: Compute λ1 numerically using λ1 = k*T/(√2 * π * d^2 * p) by plugging in the given ultra-high vacuum pressure, "
        "temperature, and representative molecular diameter. Show the full arithmetic."
    )
    N_sc = self.max_sc
    sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": sc_instruction, "context": ["user query", "thinking1", "answer1"], "agent_collaboration": "SC_CoT"}
    possible_answers2 = []
    possible_thinkings2 = []
    for i in range(N_sc):
        th2, an2 = await sc_agents[i]([taskInfo, thinking1, answer1], sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents[i].id}, thinking: {th2.content}; answer: {an2.content}")
        possible_thinkings2.append(th2)
        possible_answers2.append(an2)
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr2 = "Sub-task 2: Synthesize and choose the most consistent answer for λ1 calculation from the above agents."
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, synth_instr2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2:", sub_tasks[-1])

    # Sub-task 3: Self-Consistency Chain-of-Thought
    sc3_instruction = (
        "Sub-task 3: Retrieve or cite authoritative data for the effective electron–molecule collision cross-section σ_e-gas at 1000 kV "
        "for the dominant residual gas species. Ensure empirical sourcing."
    )
    sc3_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": sc3_instruction, "context": ["user query", "thinking1", "answer1"], "agent_collaboration": "SC_CoT"}
    possible_answers3 = []
    possible_thinkings3 = []
    for i in range(N_sc):
        th3, an3 = await sc3_agents[i]([taskInfo, thinking1, answer1], sc3_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc3_agents[i].id}, thinking: {th3.content}; answer: {an3.content}")
        possible_thinkings3.append(th3)
        possible_answers3.append(an3)
    final_decision_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr3 = "Sub-task 3: Synthesize and choose the most consistent σ_e-gas data from the above agents."
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking1, answer1] + possible_thinkings3 + possible_answers3, synth_instr3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3:", sub_tasks[-1])

    # Sub-task 4: Reflexion
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot4_instruction = (
        "Sub-task 4: Investigate and clarify the origin and physical interpretation of the factor 1.22 in the context of electron vs. molecule scattering cross-section ratios, "
        "to prevent misinterpretation as a mean free path multiplier." + reflect_inst
    )
    cot4_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot4_instruction, "context": ["user query", "thinking3", "answer3"], "agent_collaboration": "Reflexion"}
    cot4_inputs = [taskInfo, thinking3, answer3]
    thinking4, answer4 = await cot4_agent(cot4_inputs, cot4_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot4_agent.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        critic_inst4 = (
            "Please review the answer above and criticize where it might be wrong. "
            "If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
        )
        feedback4, correct4 = await critic_agent4([taskInfo, thinking4, answer4], critic_inst4, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent4.id}, feedback: {feedback4.content}; correct: {correct4.content}")
        if correct4.content == "True":
            break
        cot4_inputs += [thinking4, answer4, feedback4]
        thinking4, answer4 = await cot4_agent(cot4_inputs, cot4_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot4_agent.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4:", sub_tasks[-1])

    # Sub-task 5: Chain-of-Thought
    cot5_instruction = (
        "Sub-task 5: Using the gas density from Sub-task 2 and σ_e-gas from Sub-task 3, calculate λ2 = 1/(n·σ_e-gas) numerically. "
        "Explicitly show all steps and numerical values."
    )
    cot5_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot5_instruction, "context": ["user query", "thinking2", "answer2", "thinking3", "answer3"], "agent_collaboration": "CoT"}
    thinking5, answer5 = await cot5_agent([taskInfo, thinking2, answer2, thinking3, answer3], cot5_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot5_agent.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5:", sub_tasks[-1])

    # Sub-task 6: Debate
    debate_inst = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate6_instruction = (
        "Sub-task 6: Compare λ2 and λ1 by computing the ratio λ2/λ1, evaluate against 1.22, and rigorously debate which inequality choice matches the data. "
        + debate_inst
    )
    debate_agents6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking6 = [[] for _ in range(self.max_round)]
    all_answer6 = [[] for _ in range(self.max_round)]
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": debate6_instruction, "context": ["user query", "thinking5", "answer5"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for agent in debate_agents6:
            if r == 0:
                th6, an6 = await agent([taskInfo, thinking5, answer5], debate6_instruction, r, is_sub_task=True)
            else:
                prev_th = all_thinking6[r-1]
                prev_an = all_answer6[r-1]
                th6, an6 = await agent([taskInfo, thinking5, answer5] + prev_th + prev_an, debate6_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {th6.content}; answer: {an6.content}")
            all_thinking6[r].append(th6)
            all_answer6[r].append(an6)
    final_decision_agent6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr6 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking6, answer6 = await final_decision_agent6([taskInfo, thinking5, answer5] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Final decision." + final_instr6, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent6.id}, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6:", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs