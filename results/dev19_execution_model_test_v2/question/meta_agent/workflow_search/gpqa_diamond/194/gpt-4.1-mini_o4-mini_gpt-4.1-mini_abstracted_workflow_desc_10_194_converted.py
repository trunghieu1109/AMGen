async def forward_194(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0_1 = "Sub-task 1: Extract and transform all given physical parameters (planet radii, star radius, orbital period, impact parameter) into normalized and usable forms for orbital geometry calculations, based on the user query."
    cot_sc_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking0_1, answer0_1 = await cot_sc_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_1[i].id}, extracting and normalizing parameters, thinking: {thinking0_1.content}; answer: {answer0_1.content}")
        possible_answers_0_1.append(answer0_1)
        possible_thinkings_0_1.append(thinking0_1)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0_1, answer0_1 = await final_decision_agent_0_1([taskInfo] + possible_thinkings_0_1 + possible_answers_0_1, "Sub-task 1: Synthesize and choose the most consistent extraction and normalization of parameters.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0_1.content}; answer - {answer0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking0_1, "answer": answer0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_1_1 = "Sub-task 1: Using the extracted parameters, determine the orbital inclination of the first planet from its transit impact parameter and star radius."
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking0_1, answer0_1],
        "agent_collaboration": "SC_CoT"
    }
    thinking1_1, answer1_1 = await cot_agent_1_1([taskInfo, thinking0_1, answer0_1], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, determining orbital inclination, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc_1_1)
    print("Step 2: ", sub_tasks[-1])

    reflect_inst_1_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_2 = "Sub-task 2: Using the inclination from the first planet and the requirement that the second planet shares the same orbital plane, derive the geometric constraints on the second planet's orbital radius that allow both transit and occultation to occur." + reflect_inst_1_2
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_2 = self.max_round
    cot_inputs_1_2 = [taskInfo, thinking0_1, answer0_1, thinking1_1, answer1_1]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_reflect_instruction_1_2,
        "context": ["user query", thinking0_1, answer0_1, thinking1_1, answer1_1],
        "agent_collaboration": "Reflexion"
    }
    thinking1_2, answer1_2 = await cot_agent_1_2(cot_inputs_1_2, cot_reflect_instruction_1_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, deriving geometric constraints, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
    critic_inst_1_2 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_1_2):
        feedback_1_2, correct_1_2 = await critic_agent_1_2([taskInfo, thinking1_2, answer1_2], "Please review and provide the limitations of provided solutions." + critic_inst_1_2, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_2.id}, providing feedback, thinking: {feedback_1_2.content}; answer: {correct_1_2.content}")
        if correct_1_2.content == "True":
            break
        cot_inputs_1_2.extend([thinking1_2, answer1_2, feedback_1_2])
        thinking1_2, answer1_2 = await cot_agent_1_2(cot_inputs_1_2, cot_reflect_instruction_1_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, refining geometric constraints, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc_1_2)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_1_3 = "Sub-task 3: Apply Kepler's third law to convert the maximum allowed orbital radius of the second planet into the corresponding maximum orbital period, assuming the star's mass is solar mass."
    cot_sc_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking1_2, answer1_2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking1_3, answer1_3 = await cot_sc_agents_1_3[i]([taskInfo, thinking1_2, answer1_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_3[i].id}, converting orbital radius to period, thinking: {thinking1_3.content}; answer: {answer1_3.content}")
        possible_answers_1_3.append(answer1_3)
        possible_thinkings_1_3.append(thinking1_3)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_3, answer1_3 = await final_decision_agent_1_3([taskInfo, thinking1_2, answer1_2] + possible_thinkings_1_3 + possible_answers_1_3, "Sub-task 3: Synthesize and choose the most consistent maximum orbital period from orbital radius.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking1_3.content}; answer - {answer1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking1_3, "answer": answer1_3}
    logs.append(subtask_desc_1_3)
    print("Step 4: ", sub_tasks[-1])

    debate_instr_2_1 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_2_1 = "Sub-task 1: Evaluate the candidate maximum orbital periods against the geometric and orbital constraints to select the correct maximum orbital period of the second planet that exhibits both transit and occultation." + debate_instr_2_1
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking1_3, answer1_3],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking2_1, answer2_1 = await agent([taskInfo, thinking1_3, answer1_3], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos_2_1 = [taskInfo, thinking1_3, answer1_3] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking2_1, answer2_1 = await agent(input_infos_2_1, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating candidate periods, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
            all_thinking_2_1[r].append(thinking2_1)
            all_answer_2_1[r].append(answer2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_1, answer2_1 = await final_decision_agent_2_1([taskInfo, thinking1_3, answer1_3] + all_thinking_2_1[-1] + all_answer_2_1[-1], "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final maximum orbital period, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc_2_1)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2_1, answer2_1, sub_tasks, agents)
    return final_answer, logs
