async def forward_197(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0, Sub-task 0.1: SC-CoT to extract inputs and list species
    sc_instruction = (
        "Sub-task 0.1: Extract quantitative inputs c(Co)=1e-2 M, [SCN-]=0.1 M, beta1=9, beta2=40, "
        "beta3=63, beta4=16 and list all possible Co–SCN species."
    )
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings, possible_answers = [], []
    subtask_desc0_1 = {"subtask_id": "subtask_0.1", "instruction": sc_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking, answer = await cot_agents[i]([taskInfo], sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, extracting inputs and listing species, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_decision_0_1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0_1, answer0_1 = await final_decision_0_1([taskInfo] + possible_thinkings + possible_answers,
                                                   "Sub-task 0.1: Synthesize the extracted quantitative inputs and species list.",
                                                   is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking0_1.content}; answer - {answer0_1.content}")
    subtask_desc0_1['response'] = {"thinking": thinking0_1, "answer": answer0_1}
    logs.append(subtask_desc0_1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 0, Sub-task 0.2: Reflexion to identify blue species
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    reflect_instruction = (
        "Sub-task 0.2: Identify which Co–SCN species corresponds to the blue dithiocyanato complex. " + reflect_inst
    )
    cot_reflect = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc0_2 = {"subtask_id": "subtask_0.2", "instruction": reflect_instruction,
                       "context": ["user query","thinking0_1","answer0_1"], "agent_collaboration": "Reflexion"}
    thinking0_2, answer0_2 = await cot_reflect([taskInfo, thinking0_1, answer0_1], reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_reflect.id}, initial identification, thinking: {thinking0_2.content}; answer: {answer0_2.content}")
    for i in range(self.max_round):
        feedback, correct = await critic([taskInfo, thinking0_2, answer0_2],
                                         "Please review the answer above and criticize where it might be wrong. If absolutely sure it is correct, output exactly 'True' in 'correct'.",
                                         i, is_sub_task=True)
        agents.append(f"Critic agent {critic.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        thinking0_2, answer0_2 = await cot_reflect([taskInfo, thinking0_1, answer0_1, thinking0_2, answer0_2, feedback],
                                                   reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_reflect.id}, refined identification, thinking: {thinking0_2.content}; answer: {answer0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking0_2.content}; answer - {answer0_2.content}")
    subtask_desc0_2['response'] = {"thinking": thinking0_2, "answer": answer0_2}
    logs.append(subtask_desc0_2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 1, Sub-task 1.1: Debate to formulate equations
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction = (
        "Sub-task 1.1: Formulate the cumulative stability and mass-balance equations to derive an expression for the fraction of Co(SCN)2 among total Co species. "
        + debate_instr
    )
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                     for role in self.debate_role]
    all_think1 = [[] for _ in range(self.max_round)]
    all_ans1 = [[] for _ in range(self.max_round)]
    subtask_desc1_1 = {"subtask_id": "subtask_1.1", "instruction": debate_instruction,
                        "context": ["user query","thinking0_2","answer0_2"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking1, answer1 = await agent([taskInfo, thinking0_2, answer0_2], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking0_2, answer0_2] + all_think1[r-1] + all_ans1[r-1]
                thinking1, answer1 = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking1.content}; answer: {answer1.content}")
            all_think1[r].append(thinking1)
            all_ans1[r].append(answer1)
    final_decision_1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_1, answer1_1 = await final_decision_1(
        [taskInfo, thinking0_2, answer0_2] + all_think1[-1] + all_ans1[-1],
        "Sub-task 1.1: Given all the above thinking and answers, reason over them carefully and provide the derived equations.",
        is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc1_1)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2, Sub-task 2.1: CoT to compute numerical percentage
    cot_instruction2 = (
        "Sub-task 2.1: Insert numerical values into the derived expressions and solve for the percentage of Co(SCN)2."
    )
    cot_agent2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2_1 = {"subtask_id": "subtask_2.1", "instruction": cot_instruction2,
                        "context": ["user query","thinking1_1","answer1_1"], "agent_collaboration": "CoT"}
    thinking2_1, answer2_1 = await cot_agent2([taskInfo, thinking1_1, answer1_1], cot_instruction2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, computing numerical result, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc2_1)
    print("Step 4: ", sub_tasks[-1])

    # Stage 2, Sub-task 2.2: Reflexion to compare with choices
    reflect_inst2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to choose the closest answer choice."
    reflect_instruction2 = "Sub-task 2.2: Compare the calculated percentage with the given answer choices and select the closest match. " + reflect_inst2
    cot_reflect2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic2 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc2_2 = {"subtask_id": "subtask_2.2", "instruction": reflect_instruction2,
                       "context": ["user query","thinking2_1","answer2_1"], "agent_collaboration": "Reflexion"}
    thinking2_2, answer2_2 = await cot_reflect2([taskInfo, thinking2_1, answer2_1], reflect_instruction2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_reflect2.id}, initial choice comparison, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
    for i in range(self.max_round):
        feedback2, correct2 = await critic2([taskInfo, thinking2_2, answer2_2],
                                            "Please review the choice above and criticize where it might be wrong. If correct, output exactly 'True' in 'correct'.",
                                            i, is_sub_task=True)
        agents.append(f"Critic agent {critic2.id}, feedback: {feedback2.content}; correct: {correct2.content}")
        if correct2.content.strip() == "True":
            break
        thinking2_2, answer2_2 = await cot_reflect2([taskInfo, thinking2_1, answer2_1, thinking2_2, answer2_2, feedback2],
                                                   reflect_instruction2, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_reflect2.id}, refined comparison, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    subtask_desc2_2['response'] = {"thinking": thinking2_2, "answer": answer2_2}
    logs.append(subtask_desc2_2)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2_2, answer2_2, sub_tasks, agents)
    return final_answer, logs