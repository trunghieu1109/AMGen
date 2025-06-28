async def forward(self, taskInfo):
    from collections import Counter
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Condition Identification and Decomposition]
    
    [Objective]
    - Systematically identify all necessary conditions required to solve the query, breaking them down into clear, sequential steps.
    - Ensure each condition aligns with a distinct subtask or reasoning component, forming a logical foundation for inference in later stages.
    
    [Agent Collaborations]
    - Apply Chain-of-Thought or Self-Consistency Chain-of-Thought techniques to guide step-by-step reasoning and enhance reliability of intermediate outputs.
    - Integrate relevant context, task specifications, and outputs from prior subtasks to maintain coherence and consistency.
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 1, Subtask 2)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 1: Calculate condition #1.
    cot_instruction = "Sub-task 1: Calculate [condition #1], with context of ......."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, calculating [condition #1], thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Calculate condition #2.
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, calculate [condition #2], with context ....."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    for i in range(N):
        # each cot-agent generate a calculation for [condition #2]
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, determining [condition #2], thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
        
    # then choose the most frequent answer from potential set
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 2: Inference and Synthesis]
    
    [Objective]
    - Utilize the outputs and insights derived in Stage 1 to infer the final answer for each corresponding query.
    - Synthesize intermediate calculations and contextual understanding to generate accurate and well-supported conclusions.
    
    [Agent Collaborations]
    - Employ a Debate/Reflexion Pattern to enable critical evaluation and refinement of generated answers through multi-agent reasoning.
    - Coordinate agent contributions by cross-referencing both Stage 1 results and real-time deductions within this stage to strengthen final outputs.
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 1, Subtask 2)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 3: Calculate the final output for corresponding query
    cot_reflect_instruction = "Sub-task 3: Based on the outputs from Sub-task 1 and sub-task 2, calculate the [final answer]"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    # Input for cot-agent
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    # generate the first answer.
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, calculating [final answer], thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max):
        # use critic-agent to reflect and criticise the proposed answer from cot-agent
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], "please review [final answer] calculation and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        
        # stop if the answer is correct
        if correct.content == "True":
            break
        
        # merge recently answer and feedback from critic and then pass to cot-agent in the next round
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining [final_answer], thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer