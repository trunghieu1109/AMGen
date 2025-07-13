async def forward_28(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Extract and summarize geometric properties
    cot_instruction = "Sub-task 1: Extract and summarize the given geometric properties of the torus and sphere, including the major and minor radii of the torus and the radius of the sphere."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing geometric properties, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: Derive geometric relationship and conditions for tangency
    cot_sc_instruction = "Sub-task 2: Derive the geometric relationship and conditions for tangency between the torus and the sphere. Solve the system of equations representing the tangency conditions to find the center-to-center distances (D) for both configurations."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    possible_thinkings = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, deriving tangency conditions, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2)
        possible_thinkings.append(thinking2)
    final_instr = "Given all the above thinking and answers, find the most consistent and correct solutions for the tangency conditions."
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers + possible_thinkings, "Sub-task 2: Synthesize and choose the most consistent answer for tangency conditions." + final_instr, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 3: Clarify configurations of tangency
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 3: Clarify the configurations of tangency, specifically the internal and external tangency conditions, and how they relate to the radii of the circles of tangency." + reflect_inst
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, refining tangency configurations, thinking: {thinking3.content}; answer: {answer3.content}")
    critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], "Please review and provide the limitations of provided solutions" + critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining tangency configurations, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 4: Calculate radii of circles of tangency
    cot_instruction_4 = "Sub-task 4: Calculate the radius of the circle of tangency when the torus is internally tangent to the sphere, denoted as r_i, using the correct center-to-center distance and torus parametrization."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating r_i, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = "Sub-task 5: Calculate the radius of the circle of tangency when the torus is externally tangent to the sphere, denoted as r_o, using the correct center-to-center distance and torus parametrization."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3, answer3], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, calculating r_o, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    # Stage 5: Compute the difference r_i - r_o
    cot_sc_instruction_6 = "Sub-task 6: Compute the difference r_i - r_o and express it as a fraction in simplest form, identifying m and n as coprime integers."
    N = self.max_sc
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_6 = []
    possible_thinkings_6 = []
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking6, answer6 = await cot_agents_6[i]([taskInfo, thinking4, answer4, thinking5, answer5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, computing r_i - r_o, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6)
        possible_thinkings_6.append(thinking6)
    final_instr_6 = "Given all the above thinking and answers, find the most consistent and correct solutions for r_i - r_o."
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + possible_answers_6 + possible_thinkings_6, "Sub-task 6: Synthesize and choose the most consistent answer for r_i - r_o." + final_instr_6, is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    # Stage 6: Calculate the sum m + n
    reflect_inst_7 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_7 = "Sub-task 7: Calculate the sum m + n based on the fraction obtained in the previous subtask. Verify the coprimality of m and n before proceeding." + reflect_inst_7
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking6, answer6]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, calculating m + n, thinking: {thinking7.content}; answer: {answer7.content}")
    critic_inst_7 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_7):
        feedback_7, correct_7 = await critic_agent_7([taskInfo, thinking7, answer7], "Please review and provide the limitations of provided solutions" + critic_inst_7, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback_7.content}; answer: {correct_7.content}")
        if correct_7.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback_7])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining m + n calculation, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs