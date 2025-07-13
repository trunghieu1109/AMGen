async def forward_163(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_stage0 = "Sub-task 1: Extract and summarize all given physical parameters and assumptions from the problem statement, including orbital periods, radial velocity amplitudes, and assumptions about orbit inclination and circularity. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage0 = self.max_round
    all_thinking_stage0 = [[] for _ in range(N_max_stage0)]
    all_answer_stage0 = [[] for _ in range(N_max_stage0)]
    subtask_desc0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": debate_instr_stage0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage0):
        for i, agent in enumerate(debate_agents_stage0):
            if r == 0:
                thinking0, answer0 = await agent([taskInfo], debate_instr_stage0, r, is_sub_task=True)
            else:
                input_infos_0 = [taskInfo] + all_thinking_stage0[r-1] + all_answer_stage0[r-1]
                thinking0, answer0 = await agent(input_infos_0, debate_instr_stage0, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, extracting and summarizing data, thinking: {thinking0.content}; answer: {answer0.content}")
            all_thinking_stage0[r].append(thinking0)
            all_answer_stage0[r].append(answer0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + all_thinking_stage0[-1] + all_answer_stage0[-1], "Sub-task 1: Extract and summarize all given physical parameters and assumptions from the problem statement." + " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, extracting and summarizing data, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Stage 0 Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_ratio = "Sub-task 1: Compute the mass ratio of the two stars in each system using the ratio of their radial velocity amplitudes, based on the extracted parameters from Stage 0. Consider multiple reasoning paths and select the most consistent answer."
    cot_sc_instruction_mass = "Sub-task 2: Calculate the total mass of each binary system using the orbital period and radial velocity amplitudes, applying Kepler's third law and orbital velocity relations, based on the extracted parameters from Stage 0. Consider multiple reasoning paths and select the most consistent answer."
    N = self.max_sc
    cot_agents_ratio = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_mass = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]

    possible_answers_ratio = []
    possible_thinkings_ratio = []
    subtask_desc1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_ratio,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1_1, answer1_1 = await cot_agents_ratio[i]([taskInfo, thinking0, answer0], cot_sc_instruction_ratio, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_ratio[i].id}, computing mass ratio, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
        possible_answers_ratio.append(answer1_1)
        possible_thinkings_ratio.append(thinking1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_1, answer1_1 = await final_decision_agent_1_1([taskInfo, thinking0, answer0] + possible_thinkings_ratio + possible_answers_ratio, "Sub-task 1: Synthesize and choose the most consistent mass ratio answer." + " Given all the above thinking and answers, find the most consistent and correct solutions for the mass ratio.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing mass ratio, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc1_1)
    print("Step 1.1: ", sub_tasks[-1])

    possible_answers_mass = []
    possible_thinkings_mass = []
    subtask_desc1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_mass,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1_2, answer1_2 = await cot_agents_mass[i]([taskInfo, thinking0, answer0], cot_sc_instruction_mass, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_mass[i].id}, computing total mass, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
        possible_answers_mass.append(answer1_2)
        possible_thinkings_mass.append(thinking1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_2, answer1_2 = await final_decision_agent_1_2([taskInfo, thinking0, answer0] + possible_thinkings_mass + possible_answers_mass, "Sub-task 2: Synthesize and choose the most consistent total mass answer." + " Given all the above thinking and answers, find the most consistent and correct solutions for the total mass.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing total mass, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc1_2['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc1_2)
    print("Step 1.2: ", sub_tasks[-1])

    reflect_inst_stage2_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_1 = "Sub-task 1: Combine the mass ratio and total mass calculations to determine the sum of the masses of the two stars in each system." + reflect_inst_stage2_1
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_1 = self.max_round
    cot_inputs_2_1 = [taskInfo, thinking1_1, answer1_1, thinking1_2, answer1_2]
    subtask_desc2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking1_1, answer1_1, thinking1_2, answer1_2],
        "agent_collaboration": "Reflexion"
    }
    thinking2_1, answer2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, combining mass ratio and total mass, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    for i in range(N_max_2_1):
        feedback, correct = await critic_agent_2_1([taskInfo, thinking2_1, answer2_1], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_1.extend([thinking2_1, answer2_1, feedback])
        thinking2_1, answer2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining combined mass, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc2_1)
    print("Step 2.1: ", sub_tasks[-1])

    reflect_inst_stage2_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_2 = "Sub-task 2: Compute the factor by which system_1 is more massive than system_2 by taking the ratio of their total masses." + reflect_inst_stage2_2
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking2_1, answer2_1]
    subtask_desc2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking2_1, answer2_1],
        "agent_collaboration": "Reflexion"
    }
    thinking2_2, answer2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, computing mass factor, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
    for i in range(N_max_2_2):
        feedback, correct = await critic_agent_2_2([taskInfo, thinking2_2, answer2_2], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_2.extend([thinking2_2, answer2_2, feedback])
        thinking2_2, answer2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining mass factor, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
    sub_tasks.append(f"Stage 2 Sub-task 2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    subtask_desc2_2['response'] = {"thinking": thinking2_2, "answer": answer2_2}
    logs.append(subtask_desc2_2)
    print("Step 2.2: ", sub_tasks[-1])

    debate_instr_stage3 = "Sub-task 1: Select the answer choice that best matches the computed mass ratio factor between system_1 and system_2. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage3 = self.max_round
    all_thinking_stage3 = [[] for _ in range(N_max_stage3)]
    all_answer_stage3 = [[] for _ in range(N_max_stage3)]
    subtask_desc3 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instr_stage3,
        "context": ["user query", thinking2_2, answer2_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage3):
        for i, agent in enumerate(debate_agents_stage3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2_2, answer2_2], debate_instr_stage3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2_2, answer2_2] + all_thinking_stage3[r-1] + all_answer_stage3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instr_stage3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting best answer choice, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_stage3[r].append(thinking3)
            all_answer_stage3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2_2, answer2_2] + all_thinking_stage3[-1] + all_answer_stage3[-1], "Sub-task 1: Select the best matching answer choice." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting best answer choice, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Stage 3 Sub-task 1 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
