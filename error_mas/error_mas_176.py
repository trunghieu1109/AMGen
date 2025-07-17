async def forward_176(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs =  []

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 0: Extract and summarize all given numerical and qualitative information about Star_1 and Star_2 (radii, masses, peak wavelengths, radial velocities, black-body assumption, answer choices)." + reflect_inst
    cot_agent0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent0 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max0 = self.max_round
    thinking0, answer0 = await cot_agent0([taskInfo], cot_reflect_instruction, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent0.id}, initial extraction, thinking: {thinking0.content}; answer: {answer0.content}")
    for i in range(N_max0):
        feedback0, correct0 = await critic_agent0([taskInfo, thinking0, answer0], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", is_sub_task=True)
        agents.append(f"Critic agent {critic_agent0.id}, critique: {feedback0.content}; correct: {correct0.content}")
        if correct0.content == "True":
            break
        thinking0, answer0 = await cot_agent0([taskInfo, thinking0, answer0, feedback0], cot_reflect_instruction, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent0.id}, refined extraction, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    logs.append({"subtask_id": "subtask_0", "instruction": cot_reflect_instruction, "context": ["user query"], "agent_collaboration": "Reflexion", "response": {"thinking": thinking0, "answer": answer0})
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 1: Use Wien’s displacement law on the identical peak wavelengths to conclude that T1 = T2."
    N = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings1 = []
    possible_answers1 = []
    for agent in cot_agents1:
        thinking1_i, answer1_i = await agent([taskInfo, thinking0, answer0], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, conclude T1=T2, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo, thinking0, answer0] + possible_thinkings1 + possible_answers1, "Sub-task 1: Synthesize and choose the most consistent answer for T1=T2.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append({"subtask_id": "subtask_1", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask 0", "answer of subtask 0"], "agent_collaboration": "SC_CoT", "response": {"thinking": thinking1, "answer": answer1})
    print("Step 2: ", sub_tasks[-1])

    cot_instruction2 = "Sub-task 2: Apply the Stefan–Boltzmann law with R1/R2 = 1.5 and T1 = T2 to compute L1/L2 = (R1/R2)^2."
    cot_agent2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent2([taskInfo, thinking0, answer0, thinking1, answer1], cot_instruction2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, compute luminosity ratio, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append({"subtask_id": "subtask_2", "instruction": cot_instruction2, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "CoT", "response": {"thinking": thinking2, "answer": answer2})
    print("Step 3: ", sub_tasks[-1])

    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction3 = "Sub-task 3: Compare the computed luminosity ratio to the provided choices (2.25, 2.35, 2.32, 2.23) and assess if Doppler shifts alter the result, then select the best match." + debate_instr
    debate_agents3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max3)]
    all_answer3 = [[] for _ in range(N_max3)]
    for r in range(N_max3):
        for i, agent in enumerate(debate_agents3):
            if r == 0:
                thinking3_i, answer3_i = await agent([taskInfo, thinking2, answer2], debate_instruction3, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3_i, answer3_i = await agent(inputs, debate_instruction3, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
            all_thinking3[r].append(thinking3_i)
            all_answer3[r].append(answer3_i)
    final_decision_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: " + debate_instruction3 + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append({"subtask_id": "subtask_3", "instruction": debate_instruction3, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "Debate", "response": {"thinking": thinking3, "answer": answer3})
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs