async def forward_176(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = "Sub-task 1: Compute the Doppler factor for Star_2 with radial velocity v=700 km/s using factor = 1 + v/c"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing Doppler factor for Star_2, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)

    cot_sc_instruction = "Sub-task 2: Compute the true peak wavelength λ_peak,2 of Star_2 by dividing the observed peak wavelength by the Doppler factor"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, computing λ_peak,2, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content] = thinking2_i
        answermapping[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)

    cot_reflect_instruction = "Sub-task 3: Compute the temperature ratio T1/T2 using λ_peak,1 and λ_peak,2 from Wien's displacement law"
    cot_agent_ref = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_ref(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_ref.id}, computing T1/T2, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], "please review the temperature ratio computation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_ref(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_ref.id}, refinement round {i+1}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)

    cot_instruction4 = "Sub-task 4: Compute the squared radius ratio (R1/R2)^2 given R1/R2 = 1.5"
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction4,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent4([taskInfo], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, computing squared radius ratio, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)

    debate_instruction5 = "Sub-task 5: Compute luminosity ratio L1/L2 using (R1/R2)^2 and (T1/T2)^4"
    debate_agents5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max5)]
    all_answer5 = [[] for _ in range(N_max5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction5,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max5):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5_r, answer5_r = await agent([taskInfo, thinking3, answer3, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                input_infos5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5_r, answer5_r = await agent(input_infos5, debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing L1/L2, thinking: {thinking5_r.content}; answer: {answer5_r.content}")
            all_thinking5[r].append(thinking5_r)
            all_answer5[r].append(answer5_r)
    final_decision_agent5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Decide on the best computed luminosity ratio L1/L2.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding L1/L2, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)

    cot_instruction6 = "Sub-task 6: Select the closest option among the provided choices (~2.25, ~2.35, ~2.32, ~2.23) to the computed luminosity ratio"
    cot_agent6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot_instruction6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, selecting closest option, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6["response"] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs