async def forward_171(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: Debate to derive theoretical Boltzmann relation
    debate_instr1 = (
        "Sub-task 1: Derive the theoretical relation between the population ratio of the excited level in two stars "
        "and their effective temperatures using the Boltzmann distribution under LTE." 
        "Given solutions to the problem from other agents, consider their opinions as additional advice. "
        "Please think carefully and provide an updated answer.")
    debate_agents1 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    all_thinking1 = [[] for _ in range(self.max_round)]
    all_answer1 = [[] for _ in range(self.max_round)]
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for agent in debate_agents1:
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo], debate_instr1, r, is_sub_task=True)
            else:
                inputs = [taskInfo] + all_thinking1[r-1] + all_answer1[r-1]
                thinking_i, answer_i = await agent(inputs, debate_instr1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking1[r].append(thinking_i)
            all_answer1[r].append(answer_i)
    final_debate_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr1 = (
        "Sub-task 1: Derive Boltzmann relation." 
        "Given all the above thinking and answers, reason over them carefully and provide a final answer.")
    thinking1, answer1 = await final_debate_agent1(
        [taskInfo] + all_thinking1[-1] + all_answer1[-1], final_instr1, is_sub_task=True
    )
    agents.append(f"Final Decision agent, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: SC-CoT to compute ΔE/k_B
    cot_sc_instruction2 = (
        "Sub-task 2: Based on the derived relation, compute the numerical ratio ΔE/k_B using ΔE=1.38×10^-23 J "
        "and Boltzmann's constant.")
    N2 = self.max_sc
    cot_agents2 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N2)
    ]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2, answer2 = await cot_agents2[i](
            [taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True
        )
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thinkings2.append(thinking2)
        possible_answers2.append(answer2)
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr2 = "Sub-task 2: Synthesize and choose the most consistent computation for ΔE/k_B."
    thinking2, answer2 = await final_decision_agent2(
        [taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2,
        synth_instr2, is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision_agent2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Reflexion to substitute and simplify
    reflect_inst3 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better.")
    cot_reflect_instruction3 = (
        "Sub-task 3: Substitute the computed ΔE/k_B into the derived Boltzmann relation and simplify to express ln(2) purely in terms of T1 and T2. "
        + reflect_inst3)
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction3,
        "context": ["user query", thinking1, answer1, thinking2, answer2],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent3(inputs3, cot_reflect_instruction3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback3, correct3 = await critic_agent3(
            [taskInfo, thinking3, answer3],
            "Please review and provide the limitations of the provided solution. "
            "If you are absolutely sure it is correct, output exactly 'True' in 'correct'.",
            i, is_sub_task=True
        )
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback3.content}; correct: {correct3.content}")
        if correct3.content.strip() == "True":
            break
        inputs3 += [thinking3, answer3, feedback3]
        thinking3, answer3 = await cot_agent3(inputs3, cot_reflect_instruction3, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: SC-CoT to compare with provided choices
    cot_sc_instruction4 = (
        "Sub-task 4: Compare the simplified expression for ln(2) with each of the four provided choices and identify which one matches exactly.")
    N4 = self.max_sc
    cot_agents4 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N4)
    ]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction4,
        "context": ["user query", thinking1, answer1, thinking2, answer2, thinking3, answer3],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N4):
        thinking4_tmp, answer4_tmp = await cot_agents4[i](
            [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3],
            cot_sc_instruction4, is_sub_task=True
        )
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, thinking: {thinking4_tmp.content}; answer: {answer4_tmp.content}")
        possible_thinkings4.append(thinking4_tmp)
        possible_answers4.append(answer4_tmp)
    final_decision_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr4 = "Sub-task 4: Synthesize and choose which provided choice matches the simplified ln(2) expression exactly."
    thinking4, answer4 = await final_decision_agent4(
        [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
        + possible_thinkings4 + possible_answers4,
        synth_instr4, is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs