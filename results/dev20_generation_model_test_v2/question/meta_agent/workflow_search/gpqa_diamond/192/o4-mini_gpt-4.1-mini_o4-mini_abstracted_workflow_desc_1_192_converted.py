async def forward_192(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: SC-CoT Extract and Classify Relations
    cot_sc_instruction = "Sub-task 1: Based on the user query, extract and classify the observational relations and variables: N∝1/plx^5, plx∝1/r, fixed solid angle, per-unit-distance requirement."
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_think1 = []
    possible_ans1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N1):
        thinking1_i, answer1_i = await cot_agents1[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_think1.append(thinking1_i)
        possible_ans1.append(answer1_i)
    final_decision1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr1 = "Sub-task 1: Synthesize and choose the most consistent extraction of relations."
    thinking1, answer1 = await final_decision1([taskInfo] + possible_think1 + possible_ans1, synth_instr1, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision1.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: Reflexion Substitute plx=1/r
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 2: Substitute plx=1/r into N(plx) to derive N(r) ∝ r^5." + reflect_inst
    cot2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs2 = [taskInfo, thinking1, answer1]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_reflect_instruction,
        "context": ["user query", "result of subtask 1"],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot2(cot_inputs2, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(self.max_round):
        critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
        feedback2, correct2 = await critic([taskInfo, thinking2, answer2], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic.id}, feedback: {feedback2.content}; correct: {correct2.content}")
        if correct2.content.strip() == "True":
            break
        cot_inputs2.extend([thinking2, answer2, feedback2])
        thinking2, answer2 = await cot2(cot_inputs2, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot2.id}, revised thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 3: SC-CoT Jacobian Adjustment
    cot_sc_instruction3 = "Sub-task 3: Compute the density per unit distance by including the Jacobian factor |d(plx)/dr|=(1/r^2) to obtain N(r) ∝ r^3."  
    N3 = self.max_sc
    cot_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_think3 = []
    possible_ans3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction3,
        "context": ["user query", "result of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N3):
        thinking3_i, answer3_i = await cot_agents3[i]([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3[i].id}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_think3.append(thinking3_i)
        possible_ans3.append(answer3_i)
    final_decision3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr3 = "Sub-task 3: Synthesize and choose the most consistent density result."
    thinking3, answer3 = await final_decision3([taskInfo, thinking2, answer2] + possible_think3 + possible_ans3, synth_instr3, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 4: Debate to Select Exponent
    debate_instr = "Sub-task 4: Compare the derived N(r)∝r^3 with the provided answer choices and select the matching exponent. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_think4 = [[] for _ in range(self.max_round)]
    all_ans4 = [[] for _ in range(self.max_round)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instr,
        "context": ["user query", "result of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking4_i, answer4_i = await agent([taskInfo, thinking3, answer3], debate_instr, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3, answer3] + all_think4[r-1] + all_ans4[r-1]
                thinking4_i, answer4_i = await agent(inputs, debate_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
            all_think4[r].append(thinking4_i)
            all_ans4[r].append(answer4_i)
    final_decision4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr4 = "Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking4, answer4 = await final_decision4([taskInfo, thinking3, answer3] + all_think4[-1] + all_ans4[-1], final_instr4, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs