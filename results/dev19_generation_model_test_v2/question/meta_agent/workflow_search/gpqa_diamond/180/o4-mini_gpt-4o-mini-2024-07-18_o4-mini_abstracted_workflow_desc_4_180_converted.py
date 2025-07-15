async def forward_180(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    sc1_instruction = "Sub-task 1: Gather quantitative SSM neutrino flux data for all sources contributing in the 700–900 keV range, explicitly listing line vs. continuum contributions (e.g. 7Be line at 861 keV ≃5×10⁹ /cm²·s; 8B continuum tail ≃5×10⁶ /cm²·s)."
    N1 = self.max_sc
    sc1_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": sc1_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N1):
        thinking_i, answer_i = await sc1_agents[i]([taskInfo], sc1_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc1_agents[i].id}, gathering flux data, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings1.append(thinking_i)
        possible_answers1.append(answer_i)
    final_decision1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision1([taskInfo] + possible_thinkings1 + possible_answers1, "Sub-task 1: Synthesize and choose the most consistent flux data values for all sources in the 700–900 keV range.", is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision1.id}, synthesized flux data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    sc2_instruction = "Sub-task 2: Using the flux values from Subtask 1, integrate each source’s spectrum over Band 1 (700–800 keV) and Band 2 (800–900 keV) under steady-state conditions, producing numeric contributions per source and per band. Ensure the strong 7Be line is correctly assigned to Band 2."
    N2 = self.max_sc
    sc2_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": sc2_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking_i, answer_i = await sc2_agents[i]([taskInfo, thinking1, answer1], sc2_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc2_agents[i].id}, integrating spectra, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings2.append(thinking_i)
        possible_answers2.append(answer_i)
    final_decision2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 2: Synthesize and determine integrated flux contributions per source and per band.", is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision2.id}, computed band contributions, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 3: Simulate the sudden stoppage of the pp-III branch by subtracting only the 8B continuum contributions from both bands, while retaining all other sources intact." + reflect_inst
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent3(inputs3, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, initial subtraction simulation, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback_i, correct_i = await critic_agent3([taskInfo, thinking3, answer3], "Please review and provide the limitations of the solution. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback_i.content}; correct: {correct_i.content}")
        if correct_i.content == "True":
            break
        inputs3.extend([thinking3, answer3, feedback_i])
        thinking3, answer3 = await cot_agent3(inputs3, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refined subtraction, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot4_instruction = "Sub-task 4: Sum the remaining fluxes in Band 1 and Band 2 after the pp-III removal, then compute the ratio Flux(700–800 keV) / Flux(800–900 keV)."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot4_instruction,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot4_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, computed ratio, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instr5 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction5 = "Sub-task 5: Compare the computed ratio from Subtask 4 to the provided multiple-choice options (0.1, 10, 1, 0.01) and select the closest match. Use a structured debate to justify why the ≃10⁻² choice is correct, referencing the dominance of the 7Be line and the collapse of the 700–800 keV continuum." + debate_instr5
    debate_agents5 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                inputs5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking_i, answer_i = await agent(inputs5, debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking5[r].append(thinking_i)
            all_answer5[r].append(answer_i)
    final_decision5 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision5([taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision5.id}, final selection, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs